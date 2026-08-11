"""Process-isolated authoritative execution for natwalk sessions."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass

from .model import Cursor
from .session import Checkpoint, Session
from .sync import TreeReplica, TreeUpdate, reveal, updates
from .tree import NodeId

type CommandId = int
type CursorFactory = Callable[[], Cursor]

_DEFAULT_MAX_TREE_BYTES = 2 * 1024**3


@dataclass(frozen=True, slots=True)
class Advance:
    command_id: CommandId
    tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Rewind:
    command_id: CommandId


@dataclass(frozen=True, slots=True)
class Reveal:
    node: NodeId
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class Stop:
    pass


type Command = Advance | Rewind | Reveal | Stop


@dataclass(frozen=True, slots=True)
class TreeUpdates:
    nodes: tuple[TreeUpdate, ...]
    frontier: int


@dataclass(frozen=True, slots=True)
class EngineState:
    root: NodeId
    rewind_depth: int


@dataclass(frozen=True, slots=True)
class CommandDone:
    command_id: CommandId
    node: NodeId


@dataclass(frozen=True, slots=True)
class EngineFailed:
    exception_type: str
    message: str
    traceback: str


type Event = TreeUpdates | EngineState | CommandDone | EngineFailed


class EngineError(RuntimeError):
    """A failure raised by the authoritative engine process."""


class EngineClient:
    """Client-side process transport plus a progressive idempotent tree replica."""

    def __init__(
        self,
        factory: CursorFactory,
        *,
        max_tree_bytes: int | None = _DEFAULT_MAX_TREE_BYTES,
    ) -> None:
        if max_tree_bytes is not None and max_tree_bytes <= 0:
            raise ValueError("max_tree_bytes must be positive or None")

        context = mp.get_context("spawn")
        self._commands = context.Queue()
        self._events = context.Queue()
        self._process = context.Process(
            target=_run_engine,
            args=(factory, self._commands, self._events, max_tree_bytes),
            name="natwalk-engine",
            daemon=True,
        )
        self.replica = TreeReplica()
        self.root: NodeId | None = None
        self.rewind_depth = 0
        self.frontier = 0
        self._done: dict[CommandId, CommandDone] = {}
        self._next_command = 0
        self._failure: EngineFailed | None = None
        self._reveal_targets: dict[NodeId, int] = {}

    def __enter__(self) -> EngineClient:
        self.start()
        self.wait_ready()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def tree(self):
        tree = self.replica.tree
        if tree is None:
            raise RuntimeError("engine has not published its root yet")
        return tree

    @property
    def alive(self) -> bool:
        return self._process.is_alive()

    @property
    def ready(self) -> bool:
        return self.root is not None and self.replica.tree is not None

    def start(self) -> None:
        if self._process.pid is None:
            self._process.start()

    def wait_ready(self, timeout: float | None = None) -> None:
        """Wait for the initial root tree and state to arrive from the engine."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.ready:
            self.poll()
            if self.ready:
                return
            if self._process.pid is not None and not self.alive:
                self.poll()
                raise EngineError("engine exited before publishing its initial state")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("engine did not become ready before timeout")
            time.sleep(0.01)

    def command_id(self) -> CommandId:
        command_id = self._next_command
        self._next_command += 1
        return command_id

    def send(self, command: Command) -> None:
        self._raise_failure()
        self._commands.put(command)

    def advance(self, tokens: tuple[int, ...]) -> CommandId:
        command_id = self.command_id()
        self.send(Advance(command_id, tokens))
        return command_id

    def rewind(self) -> CommandId:
        command_id = self.command_id()
        self.send(Rewind(command_id))
        return command_id

    def reveal(self, node: NodeId, stop: int) -> None:
        """Request a larger concrete ranked prefix for one replica node."""
        distribution = self.tree[node].distribution
        start = distribution.revealed
        stop = min(stop, len(distribution))
        pending = self._reveal_targets.get(node, start)
        if stop <= max(start, pending):
            return
        self._reveal_targets[node] = stop
        self.send(Reveal(node, start, stop))

    def poll(self) -> int:
        """Apply every currently queued engine event and return the count."""
        count = 0
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            self._apply(event)
            count += 1
        self._raise_failure()
        return count

    def done(self, command_id: CommandId) -> CommandDone | None:
        self.poll()
        return self._done.get(command_id)

    def take_done(self, command_id: CommandId) -> CommandDone | None:
        """Return and forget a completed command, if its result has arrived."""
        self.poll()
        return self._done.pop(command_id, None)

    def request_stop(self) -> None:
        if self.alive:
            self._commands.put(Stop())

    def join(self, timeout: float | None = None) -> None:
        if self._process.pid is not None:
            self._process.join(timeout)

    def terminate(self) -> None:
        """Kill the engine immediately, including a currently-running model call."""
        if self.alive:
            self._process.terminate()
        self.join()

    def close(self) -> None:
        """Request cooperative shutdown, then wait for the current model call."""
        self.request_stop()
        self.join()

    def _apply(self, event: Event) -> None:
        if isinstance(event, TreeUpdates):
            self.replica.apply_many(event.nodes)
            self.frontier = event.frontier
            for update in event.nodes:
                distribution = self.tree[update.node].distribution
                target = self._reveal_targets.get(update.node)
                if target is not None and distribution.revealed >= target:
                    del self._reveal_targets[update.node]
        elif isinstance(event, EngineState):
            self.root = event.root
            self.rewind_depth = event.rewind_depth
        elif isinstance(event, CommandDone):
            self._done[event.command_id] = event
        elif isinstance(event, EngineFailed):
            self._failure = event
        else:
            raise TypeError(type(event))

    def _raise_failure(self) -> None:
        failure = self._failure
        if failure is not None:
            raise EngineError(
                f"engine {failure.exception_type}: {failure.message}\n{failure.traceback}"
            )


def _run_engine(factory: CursorFactory, commands, events, max_tree_bytes: int | None) -> None:
    """Own one Session and service commands between synchronous search discoveries."""
    try:
        session = Session(factory())
        history: list[Checkpoint] = []
        published = 0
        completed: dict[CommandId, NodeId] = {}

        def publish_tree() -> None:
            nonlocal published
            batch = updates(session.tree, start=published)
            events.put(TreeUpdates(batch, len(session.search.frontier)))
            published = len(session.tree.nodes)

        def publish_state() -> None:
            events.put(EngineState(session.root, len(history)))

        def handle(command: Command) -> bool:
            if isinstance(command, Stop):
                return False

            if isinstance(command, Reveal):
                events.put(
                    TreeUpdates(
                        (reveal(session.tree, command.node, command.start, command.stop),),
                        len(session.search.frontier),
                    )
                )
                return True

            previous = completed.get(command.command_id)
            if previous is not None:
                events.put(CommandDone(command.command_id, previous))
                publish_state()
                return True

            if isinstance(command, Advance):
                for token in command.tokens:
                    history.append(session.checkpoint())
                    session.commit((token,))
                node = session.root
            elif isinstance(command, Rewind):
                if history:
                    session.restore(history.pop())
                node = session.root
            else:
                raise TypeError(type(command))

            completed[command.command_id] = node
            publish_tree()
            publish_state()
            events.put(CommandDone(command.command_id, node))
            return True

        def can_search() -> bool:
            if max_tree_bytes is None:
                return True
            # The UI replica now retains only small revealed prefixes, so this
            # soft cap is dominated by the authoritative backend distributions.
            return session.tree.storage_bytes < max_tree_bytes

        publish_tree()
        publish_state()

        while True:
            handled = False
            while True:
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    break
                handled = True
                if not handle(command):
                    return

            if handled:
                continue

            if session.search.frontier and can_search():
                session.search.discover()
                publish_tree()
                continue

            if not handle(commands.get()):
                return
    except BaseException as exc:
        events.put(
            EngineFailed(
                exception_type=type(exc).__name__,
                message=str(exc),
                traceback="".join(traceback.format_exception(exc)),
            )
        )
