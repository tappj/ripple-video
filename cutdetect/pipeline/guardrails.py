"""Small admission limits for the public, single-instance demo."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from http import HTTPStatus

from cutdetect.pipeline.runway_client import PipelineError


class AdmissionError(PipelineError):
    """An expensive request was rejected before starting work."""

    def __init__(self, message: str, status: HTTPStatus) -> None:
        super().__init__(message)
        self.status = status


class RequestGate:
    """Bound mutation traffic, including clients that continually change identity."""

    def __init__(
        self,
        *,
        global_limit: int = 120,
        device_limit: int = 30,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._global_limit = global_limit
        self._device_limit = device_limit
        self._clock = clock
        self._started = clock()
        self._count = 0
        self._devices: dict[str, int] = {}
        self._lock = threading.Lock()

    def check(self, device: str) -> None:
        with self._lock:
            now = self._clock()
            if now - self._started >= 60:
                self._started = now
                self._count = 0
                self._devices.clear()
            count = self._devices.get(device, 0)
            if self._count >= self._global_limit or count >= self._device_limit:
                raise AdmissionError(
                    "Too many requests. Please wait a minute and try again.",
                    HTTPStatus.TOO_MANY_REQUESTS,
                )
            self._count += 1
            self._devices[device] = count + 1
