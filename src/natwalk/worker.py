"""Background execution policy for the synchronous search kernel."""

from __future__ import annotations

import threading
from contextlib import contextmanager

from .search import Search


class SearchWorker:
    """Advance ``Search.step()`` in one background thread.

    Search semantics remain entirely in :class:`Search`. This wrapper only
    serializes mutation and transports worker failures back to the foreground.
    Foreground access has priority between search steps so a fast producer
    cannot starve interactive reads and mutations.
    """

    def __init__(self, search: Search) -> None:
        self.search = search
        self._condition = threading.Condition()
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

    @contextmanager
    def access(self):
        """Serialize one foreground read/mutation with background search."""
        self._foreground_waiting.set()
        try:
            with self._condition:
                self._raise_error()
                try:
                    yield
                finally:
                    self._condition.notify_all()
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

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()

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
                    self.search.step()
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._stop = True
                self._condition.notify_all()
