"""Application-level coordination for asynchronous GUI shutdown.

The coordinator deliberately does not own worker-specific cancellation logic.
Instead, each participant exposes the smallest useful lifecycle contract:
report whether it is still active and request a cooperative shutdown.  The
window can therefore wait through Qt's event loop instead of blocking it with
``wait()`` / ``waitForFinished()`` calls.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from PySide6.QtCore import QObject, QTimer, Signal


class ShutdownParticipant(Protocol):
    """Minimal contract implemented by an application shutdown participant."""

    key: str
    label: str

    def is_active(self) -> bool:
        """Return whether shutdown must still wait for this participant."""

    def request_shutdown(self) -> None:
        """Request cancellation without blocking the GUI thread."""


@dataclass(frozen=True)
class CallbackShutdownParticipant:
    """Adapt existing workers/controllers to :class:`ShutdownParticipant`."""

    key: str
    label: str
    active_callback: Callable[[], bool]
    shutdown_callback: Callable[[], None]

    def is_active(self) -> bool:
        return bool(self.active_callback())

    def request_shutdown(self) -> None:
        self.shutdown_callback()


class ShutdownCoordinator(QObject):
    """Cancel registered work and emit ``settled`` once every task terminates.

    ``stalled`` is informational: reaching the deadline never drops worker
    references or pretends shutdown completed.  Polling continues and
    ``settled`` is emitted later if the remaining tasks really finish.
    """

    settled = Signal()
    stalled = Signal(object)  # tuple[str, ...] user-facing task labels
    cancellation_failed = Signal(str, str)  # participant label, probe/stop error

    def __init__(self, parent: QObject | None = None, *, poll_interval_ms: int = 50):
        super().__init__(parent)
        self._participants: dict[str, ShutdownParticipant] = {}
        self._in_progress = False
        self._stalled_reported = False
        self._requested_active_keys: set[str] = set()
        self._cancellation_failed_keys: set[str] = set()
        self._probe_failed_keys: set[str] = set()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(max(1, int(poll_interval_ms)))
        self._poll_timer.timeout.connect(self.check_now)

        self._deadline_timer = QTimer(self)
        self._deadline_timer.setSingleShot(True)
        self._deadline_timer.timeout.connect(self._report_stalled)

    @property
    def in_progress(self) -> bool:
        return self._in_progress

    def register(self, participant: ShutdownParticipant) -> None:
        """Register one uniquely keyed participant."""
        key = str(participant.key or "").strip()
        if not key:
            raise ValueError("shutdown participant key must not be empty")
        if key in self._participants:
            raise ValueError(f"duplicate shutdown participant: {key}")
        self._participants[key] = participant

    def register_callbacks(
        self,
        key: str,
        label: str,
        *,
        is_active: Callable[[], bool],
        request_shutdown: Callable[[], None],
    ) -> None:
        self.register(
            CallbackShutdownParticipant(
                key=key,
                label=label,
                active_callback=is_active,
                shutdown_callback=request_shutdown,
            )
        )

    def active_labels(self) -> tuple[str, ...]:
        """Return active participant labels in stable registration order."""
        return tuple(participant.label for participant in self._active_participants())

    def _active_participants(self) -> tuple[ShutdownParticipant, ...]:
        active_participants: list[ShutdownParticipant] = []
        for participant in self._participants.values():
            try:
                active = participant.is_active()
            except (RuntimeError, TypeError):
                # A Qt wrapper may disappear between completion and polling.
                self._probe_failed_keys.discard(participant.key)
                active = False
            except Exception as exc:  # noqa: BLE001
                # Fail closed for unexpected probe errors: do not declare a
                # possibly-running task settled merely because inspection failed.
                if self._in_progress and participant.key not in self._probe_failed_keys:
                    self.cancellation_failed.emit(participant.label, str(exc))
                    self._probe_failed_keys.add(participant.key)
                active = True
            else:
                self._probe_failed_keys.discard(participant.key)
            if active:
                active_participants.append(participant)
        return tuple(active_participants)

    def _request_participants(
        self,
        participants: tuple[ShutdownParticipant, ...],
    ) -> None:
        for participant in participants:
            try:
                participant.request_shutdown()
            except Exception as exc:
                if participant.key not in self._cancellation_failed_keys:
                    self.cancellation_failed.emit(participant.label, str(exc))
                self._cancellation_failed_keys.add(participant.key)
                self._requested_active_keys.discard(participant.key)
                continue
            self._cancellation_failed_keys.discard(participant.key)
            self._requested_active_keys.add(participant.key)

    def begin(self, *, timeout_ms: int = 10_000) -> bool:
        """Request shutdown for active work without waiting synchronously.

        Returns ``False`` when a shutdown pass is already in progress.
        """
        if self._in_progress:
            return False
        self._in_progress = True
        self._stalled_reported = False
        self._requested_active_keys = set()
        self._cancellation_failed_keys = set()
        self._probe_failed_keys = set()

        self._request_participants(self._active_participants())

        if not self.active_labels():
            self._finish()
            return True

        self._poll_timer.start()
        self._deadline_timer.start(max(1, int(timeout_ms)))
        return True

    def check_now(self) -> None:
        """Poll once; public so deterministic unit tests need no event loop."""
        if not self._in_progress:
            return
        active = self._active_participants()
        active_keys = {participant.key for participant in active}
        self._requested_active_keys.intersection_update(active_keys)
        self._cancellation_failed_keys.intersection_update(active_keys)
        newly_active = tuple(
            participant
            for participant in active
            if participant.key not in self._requested_active_keys
        )
        if newly_active:
            self._request_participants(newly_active)
            active = self._active_participants()
        if not active:
            self._finish()

    def _report_stalled(self) -> None:
        if not self._in_progress or self._stalled_reported:
            return
        labels = self.active_labels()
        if not labels:
            self._finish()
            return
        self._stalled_reported = True
        self.stalled.emit(labels)

    def _finish(self) -> None:
        if not self._in_progress:
            return
        self._poll_timer.stop()
        self._deadline_timer.stop()
        self._in_progress = False
        self._stalled_reported = False
        self._requested_active_keys = set()
        self._cancellation_failed_keys = set()
        self._probe_failed_keys = set()
        self.settled.emit()
