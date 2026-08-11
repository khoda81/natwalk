"""Process-isolated authoritative execution for natwalk sessions."""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from collections.abc import Callable
from dataclasses import dataclass

from .model import Cursor
from .session import Checkpoint, Session
from .sync import NodeUpdate, TreeReplica, updates
from .tree import NodeId

type CommandId = int
type CursorFactory = Callable[[], Cursor]


@dataclass(frozen=True, slots=True)
class Inspect:
    command_id: CommandId
    parent: NodeId
    rank: int


@dataclass(frozen=True, slots=True)
class Commit:
    command_id: CommandId
    tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Undo:
    command_id: CommandId


@dataclass(frozen=True, slots=True)
class Stop:
    pass


type Command = Inspect | Commit | Undo | Stop


@dataclass(frozen=True, slots=True)
class TreeUpdates:
    nodes: tuple[NodeUpdate, ...]
    frontier: int


@dataclass(frozen=True, slots=True)
class EngineState:
    root: NodeId
    history_depth: int


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
    """Client-side process transport plus an idempotent tree replica."""

    def __init__(self, factory: CursorFactory) -> None:
        context = mp.get_context("spawn")
        self._commands = context.Queue()
        self._events = context.Queue()
        self._process = context.Process(
            target=_run_engine,
            args=(factory, self._commands, self._events),
            name="natwalk-engine",
            daemon=True,
        )
        self.replica = TreeReplica()
        self.root: NodeId | None = None
        self.history_depth = 0
        self.frontier = 0
        self._done: dict[CommandId, CommandDone] = {}
        self._next_command = 0
        self._failure: EngineFailed | None = None

    def __enter__(self) -> EngineClient:
        self.start()
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

    def start(self) -> None:
        if self._process.pid is None:
            self._process.start()

    def command_id(self) -> CommandId:
        command_id = self._next_command
        self._next_command += 1
        return command_id

    def send(self, command: Inspect | Commit | Undo) -> None:
        self._raise_failure()
        self._commands.put(command)

    def inspect(self, parent: NodeId, rank: int) -> CommandId:
        command_id = self.command_id()
        self.send(Inspect(command_id, parent, rank))
        return command_id

    def commit(self, tokens: tuple[int, ...]) -> CommandId:
        command_id = self.command_id()
        self.send(Commit(command_id, tokens))
        return command_id

    def undo(self) -> CommandId:
        command_id = self.command_id()
        self.send(Undo(command_id))
        return command_id

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
        elif isinstance(event, EngineState):
            self.root = event.root
            self.history_depth = event.history_depth
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


def _run_engine(factory: CursorFactory, commands, events) -> None:
    """Own one Session and service commands between synchronous search steps."""
    try:
        session = Session(factory())
        history: list[Checkpoint] = []
        published = 0
        completed: dict[CommandId, NodeId] = {}

        def publish_tree() -> None:
            nonlocal published
            batch = updates(session.tree, start=published)
            if batch:
                events.put(TreeUpdates(batch, len(session.search.frontier)))
                published += len(batch)

        def publish_state() -> None:
            events.put(EngineState(session.root, len(history)))

        def handle(command: Command) -> bool:
            if isinstance(command, Stop):
                return False

            previous = completed.get(command.command_id)
            if previous is not None:
                events.put(CommandDone(command.command_id, previous))
                publish_state()
                return True

            if isinstance(command, Inspect):
                node = session.inspect_child(command.parent, command.rank)
            elif isinstance(command, Commit):
                if command.tokens:
                    history.append(session.checkpoint())
                    node = session.commit(command.tokens)
                else:
                    node = session.root
            elif isinstance(command, Undo):
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

            if session.search.frontier:
                session.search.step()
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
