"""Background execution policy for the synchronous search kernel."""

from __future__ import annotations

import threading
from contextlib import contextmanager

from .search import Search


class SearchWorker:
    """Advance ``Search.step()`` in one background thread.

    Search semantics remain entirely in :class:`Search`. This wrapper only
    serializes mutation so callers can inspect or change session state between
    model evaluations without maintaining a second search state machine.
    """

    def __init__(self, search: Search) -> None:
        self.search = search
        self._condition = threading.Condition()
        self._stop = False
        self._thread: threading.Thread | None = None

    def __enter__(self) -> SearchWorker:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @contextmanager
    def access(self):
        """Serialize one foreground read/mutation with background search."""
        with self._condition:
            yield
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
        while True:
            with self._condition:
                while not self._stop and not self.search.frontier:
                    self._condition.wait()
                if self._stop:
                    return
                self.search.step()
                self._condition.notify_all()
