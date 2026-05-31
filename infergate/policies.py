from __future__ import annotations

import math
from dataclasses import dataclass

from .schemas import Action, CacheState, Decision, LoadSnapshot, PolicySettings, QueueState, RequestContext


def slo_success_probability(request: RequestContext, load: LoadSnapshot, queue: QueueState) -> float:
    if request.deadline_ms is None:
        return 1.0
    estimated_queue_ms = queue.waiting_requests * 20 + load.num_requests_waiting * 25
    estimated_service_ms = request.estimated_cost * 0.6
    slack_ms = request.deadline_ms - estimated_queue_ms - estimated_service_ms
    return max(0.05, min(1.0, 1.0 / (1.0 + math.exp(-slack_ms / 1000.0))))


def session_progress_bonus(request: RequestContext) -> float:
    return 1.0 + 0.5 * request.session_progress


def cache_reuse_gain(cache: CacheState, include_cache: bool = True) -> float:
    if not include_cache:
        return 1.0
    reuse_gain = min(1.0, cache.predicted_reuse_count * 0.25)
    hit_gain = min(0.5, cache.prefix_hit_count * 0.1)
    return 1.0 + reuse_gain + hit_gain


def infergate_score(
    request: RequestContext,
    load: LoadSnapshot,
    queue: QueueState,
    cache: CacheState,
    include_cache: bool = True,
) -> float:
    return (
        request.utility
        * slo_success_probability(request, load, queue)
        * session_progress_bonus(request)
        * cache_reuse_gain(cache, include_cache=include_cache)
        / max(request.estimated_cost, 1)
    )


@dataclass
class BasePolicy:
    settings: PolicySettings
    name: str = "base"

    def decide(
        self,
        request: RequestContext,
        load_snapshot: LoadSnapshot,
        queue_state: QueueState,
        cache_state: CacheState,
    ) -> Decision:
        raise NotImplementedError

    def _queue_full(self, queue_state: QueueState) -> bool:
        return queue_state.waiting_requests >= self.settings.max_queue_size

    def _priority_fcfs(self, request: RequestContext) -> float:
        return -request.created_at


class FCFSPolicy(BasePolicy):
    name = "fcfs"

    def decide(self, request: RequestContext, load_snapshot: LoadSnapshot, queue_state: QueueState, cache_state: CacheState) -> Decision:
        if self._queue_full(queue_state):
            return Decision(action=Action.REJECT, priority=self._priority_fcfs(request), reason="queue_full")
        if queue_state.active_requests >= queue_state.max_active_requests or load_snapshot.num_requests_waiting > 0:
            return Decision(action=Action.DEFER, priority=self._priority_fcfs(request), reason="fcfs_queue")
        return Decision(action=Action.ACCEPT, priority=self._priority_fcfs(request), reason="fcfs_accept")


class SJFPolicy(BasePolicy):
    name = "sjf"

    def decide(self, request: RequestContext, load_snapshot: LoadSnapshot, queue_state: QueueState, cache_state: CacheState) -> Decision:
        priority = -float(request.estimated_cost)
        if self._queue_full(queue_state):
            return Decision(action=Action.REJECT, priority=priority, reason="queue_full")
        if load_snapshot.is_overloaded or queue_state.active_requests >= queue_state.max_active_requests:
            return Decision(action=Action.DEFER, priority=priority, reason="sjf_queue")
        return Decision(action=Action.ACCEPT, priority=priority, reason="sjf_accept")


class EDFPolicy(BasePolicy):
    name = "edf"

    def decide(self, request: RequestContext, load_snapshot: LoadSnapshot, queue_state: QueueState, cache_state: CacheState) -> Decision:
        priority = 0.0 if request.deadline_ms is None else 1_000_000.0 / max(1, request.deadline_ms)
        if self._queue_full(queue_state):
            return Decision(action=Action.REJECT, priority=priority, reason="queue_full")
        if load_snapshot.is_overloaded or queue_state.active_requests >= queue_state.max_active_requests:
            return Decision(action=Action.DEFER, priority=priority, reason="edf_queue")
        return Decision(action=Action.ACCEPT, priority=priority, reason="edf_accept")


class StaticThresholdPolicy(BasePolicy):
    name = "static_threshold"

    def decide(self, request: RequestContext, load_snapshot: LoadSnapshot, queue_state: QueueState, cache_state: CacheState) -> Decision:
        if load_snapshot.kv_cache_usage_perc >= self.settings.kv_reject_threshold:
            return Decision(action=Action.REJECT, priority=self._priority_fcfs(request), reason="kv_threshold")
        if queue_state.waiting_requests >= self.settings.waiting_reject_threshold:
            return Decision(action=Action.REJECT, priority=self._priority_fcfs(request), reason="waiting_threshold")
        if queue_state.active_requests >= queue_state.max_active_requests:
            return Decision(action=Action.DEFER, priority=self._priority_fcfs(request), reason="capacity_queue")
        return Decision(action=Action.ACCEPT, priority=self._priority_fcfs(request), reason="threshold_accept")


class VTCInspiredPolicy(BasePolicy):
    name = "vtc_inspired"

    def decide(self, request: RequestContext, load_snapshot: LoadSnapshot, queue_state: QueueState, cache_state: CacheState) -> Decision:
        tenant_debt = queue_state.tenant_token_debt.get(request.tenant_id, 0)
        priority = -float(tenant_debt + request.estimated_cost)
        if tenant_debt > self.settings.vtc_tenant_token_limit and load_snapshot.is_overloaded:
            return Decision(action=Action.REJECT, priority=priority, reason="tenant_token_limit")
        if queue_state.active_requests >= queue_state.max_active_requests or load_snapshot.is_overloaded:
            return Decision(action=Action.DEFER, priority=priority, reason="fair_queue")
        return Decision(action=Action.ACCEPT, priority=priority, reason="fair_accept")


class InferGateAdmissionPolicy(BasePolicy):
    name = "infergate_admission"
    include_cache = False

    def decide(self, request: RequestContext, load_snapshot: LoadSnapshot, queue_state: QueueState, cache_state: CacheState) -> Decision:
        score = infergate_score(request, load_snapshot, queue_state, cache_state, include_cache=self.include_cache)
        overloaded = load_snapshot.is_overloaded or queue_state.saturation >= 1.0
        if self._queue_full(queue_state):
            if score < self.settings.admission_degrade_score:
                return Decision(action=Action.REJECT, score=score, priority=score, reason="queue_full_low_score")
            return Decision(
                action=Action.DEGRADE,
                score=score,
                priority=score,
                reason="queue_full_degrade",
                degrade_max_tokens=self.settings.degraded_max_tokens,
            )
        if overloaded and score < self.settings.admission_reject_score:
            return Decision(action=Action.REJECT, score=score, priority=score, reason="low_score_overload")
        if overloaded and score < self.settings.admission_degrade_score:
            return Decision(
                action=Action.DEGRADE,
                score=score,
                priority=score,
                reason="degrade_low_score_overload",
                degrade_max_tokens=self.settings.degraded_max_tokens,
            )
        if overloaded:
            return Decision(action=Action.DEFER, score=score, priority=score, reason="high_score_queue")
        return Decision(action=Action.ACCEPT, score=score, priority=score, reason="score_accept")


class InferGateCachePolicy(InferGateAdmissionPolicy):
    name = "infergate_cache"
    include_cache = True

    def decide(self, request: RequestContext, load_snapshot: LoadSnapshot, queue_state: QueueState, cache_state: CacheState) -> Decision:
        decision = super().decide(request, load_snapshot, queue_state, cache_state)
        if cache_state.predicted_reuse_count >= 2 and load_snapshot.warmup_allowed:
            decision.warmup_candidate = True
            if decision.reason:
                decision.reason += "+warmup_candidate"
        return decision


POLICY_CLASSES = {
    "fcfs": FCFSPolicy,
    "sjf": SJFPolicy,
    "edf": EDFPolicy,
    "static_threshold": StaticThresholdPolicy,
    "vtc_inspired": VTCInspiredPolicy,
    "infergate_admission": InferGateAdmissionPolicy,
    "infergate_cache": InferGateCachePolicy,
}


def make_policy(name: str, settings: PolicySettings | None = None) -> BasePolicy:
    normalized = name.strip().lower()
    if normalized not in POLICY_CLASSES:
        raise KeyError(f"unknown policy: {name}")
    return POLICY_CLASSES[normalized](settings or PolicySettings())

