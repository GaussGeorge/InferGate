from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .schemas import QueueState


@dataclass
class QueueTicket:
    request_id: str
    priority: float
    created_at: float = field(default_factory=time.perf_counter)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    granted: bool = False


class AdmissionQueue:
    def __init__(self, max_active_requests: int = 8) -> None:
        self.max_active_requests = max_active_requests
        self._active_requests = 0
        self._pending: list[QueueTicket] = []
        self._lock = asyncio.Lock()

    async def acquire_or_queue(
        self,
        request_id: str,
        priority: float,
        timeout_ms: int,
        force_queue: bool = False,
    ) -> float:
        async with self._lock:
            if not force_queue and self._active_requests < self.max_active_requests and not self._pending:
                self._active_requests += 1
                return 0.0
            ticket = QueueTicket(request_id=request_id, priority=priority)
            self._pending.append(ticket)
            self._grant_ready_locked()
        try:
            await asyncio.wait_for(ticket.event.wait(), timeout=max(0.001, timeout_ms / 1000.0))
            return (time.perf_counter() - ticket.created_at) * 1000
        except asyncio.TimeoutError:
            async with self._lock:
                self._pending = [item for item in self._pending if item is not ticket]
                self._grant_ready_locked()
            raise

    async def release(self) -> None:
        async with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._grant_ready_locked()

    async def snapshot(self, tenant_token_debt: dict[str, int] | None = None) -> QueueState:
        async with self._lock:
            return QueueState(
                waiting_requests=len(self._pending),
                active_requests=self._active_requests,
                max_active_requests=self.max_active_requests,
                tenant_token_debt=dict(tenant_token_debt or {}),
            )

    def snapshot_nowait(self, tenant_token_debt: dict[str, int] | None = None) -> QueueState:
        return QueueState(
            waiting_requests=len(self._pending),
            active_requests=self._active_requests,
            max_active_requests=self.max_active_requests,
            tenant_token_debt=dict(tenant_token_debt or {}),
        )

    def _grant_ready_locked(self) -> None:
        self._pending.sort(key=lambda item: (-item.priority, item.created_at))
        while self._active_requests < self.max_active_requests and self._pending:
            ticket = self._pending.pop(0)
            ticket.granted = True
            self._active_requests += 1
            ticket.event.set()
