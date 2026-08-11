"""Background execution policy for the synchronous search kernel."""

from __future__ import annotations

import threading
from contextlib import contextmanager

from .search import Search


class SearchWorker:
    """Advance ``Search.step()`` in one background thread.

    The condition protects only scheduling state; it is never held while model
    evaluation runs. The mutation lock serializes search/cursor mutation with
    foreground commands. This separation makes stop requests immediate even if
    one model call is currently uninterruptible.
    """

    def __init__(self, search: Search) -> None:
        self.search = search
        self._condition = threading.Condition()
        self._mutation = threading.Lock()
        self._foreground_waiting = threading.Event()
        self._stop = False
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> SearchWorker:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _raise_error(self) -> None:
        if self._error is not None:
            raise self._error

    def raise_if_failed(self) -> None:
        """Surface a worker failure without waiting for the mutation lock."""
        self._raise_error()

    @contextmanager
    def access(self):
        """Serialize one foreground mutation with background search."""
        self._foreground_waiting.set()
        try:
            with self._mutation:
                self._raise_error()
                yield
                self._raise_error()
        finally:
            self._foreground_waiting.clear()
            with self._condition:
                self._condition.notify_all()

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="natwalk-search",
                daemon=True,
            )
            self._thread.start()

    def request_stop(self) -> None:
        """Prevent another search step without waiting for the current one."""
        with self._condition:
            self._stop = True
            self._condition.notify_all()

    def join(self) -> None:
        """Wait until the worker has finished its current search step and stopped."""
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def close(self) -> None:
        """Request stop and wait for deterministic worker shutdown."""
        self.request_stop()
        self.join()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._stop and (
                        not self.search.frontier or self._foreground_waiting.is_set()
                    ):
                        self._condition.wait()
                    if self._stop:
                        return

                with self._mutation:
                    if self._stop or self._foreground_waiting.is_set():
                        continue
                    if not self.search.frontier:
                        continue
                    self.search.step()

                with self._condition:
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._stop = True
                self._condition.notify_all()
