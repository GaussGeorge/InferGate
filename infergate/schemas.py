from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Action(str, Enum):
    ACCEPT = "accept"
    DEFER = "defer"
    DEGRADE = "degrade"
    REJECT = "reject"


class Decision(BaseModel):
    action: Action
    score: float = 0.0
    priority: float = 0.0
    reason: str = ""
    degrade_max_tokens: int | None = None
    warmup_candidate: bool = False


class RequestContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex}")
    session_id: str = "default"
    tenant_id: str = "default"
    session_step: int = 0
    session_total_steps: int = 1
    utility: float = 1.0
    deadline_ms: int | None = None
    cache_key: str
    estimated_cost: int
    prompt_tokens: int
    max_tokens: int
    prompt_text: str
    tokenizer_fallback: bool = True
    created_at: float = Field(default_factory=time.time)

    @property
    def session_progress(self) -> float:
        if self.session_total_steps <= 1:
            return 0.0
        return min(1.0, max(0.0, self.session_step / (self.session_total_steps - 1)))


class LoadSnapshot(BaseModel):
    num_requests_waiting: int = 0
    num_requests_running: int = 0
    kv_cache_usage_perc: float = 0.0
    gpu_cache_usage_perc: float = 0.0
    metrics_available: bool = True
    warmup_allowed: bool = True

    @property
    def is_overloaded(self) -> bool:
        return self.num_requests_waiting > 0 or self.kv_cache_usage_perc >= 0.85


class QueueState(BaseModel):
    waiting_requests: int = 0
    active_requests: int = 0
    max_active_requests: int = 8
    tenant_token_debt: dict[str, int] = Field(default_factory=dict)

    @property
    def saturation(self) -> float:
        return min(1.0, self.active_requests / max(1, self.max_active_requests))


class CacheState(BaseModel):
    cache_key: str
    cache_backend: str = "vllm_apc"
    seen_count: int = 0
    prefix_hit_count: int = 0
    predicted_reuse_count: int = 0
    utility_sum: float = 0.0
    total_prompt_tokens: int = 0
    last_seen_ts: float | None = None
    last_hit_ts: float | None = None


class PolicySettings(BaseModel):
    max_queue_size: int = 128
    max_active_requests: int = 8
    queue_timeout_ms: int = 120_000
    kv_reject_threshold: float = 0.92
    waiting_reject_threshold: int = 64
    admission_reject_score: float = 0.0010
    admission_degrade_score: float = 0.0020
    degraded_max_tokens: int = 128
    vtc_tenant_token_limit: int = 60_000
    warmup_budget_fraction: float = 0.10


class TraceRecord(BaseModel):
    request_id: str
    session_id: str
    policy: str
    decision: str
    score: float | None = None
    reason: str | None = None
    estimated_cost: int
    utility: float
    session_step: int
    queue_wait_ms: float
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    gateway_ms: float | None = None
    prompt_tokens: int
    completion_tokens: int = 0
    accepted: bool = False
    rejected: bool = False
    degraded: bool = False
    step0_rejection: bool = False
    cache_key: str | None = None
    cache_backend: str = "vllm_apc"
    tokenizer_fallback: bool = False
    error: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
