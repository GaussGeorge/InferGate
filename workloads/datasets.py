from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    long_context_ratio: float
    session_ratio: float
    shared_prefix_ratio: float
    utility_skew: float


WORKLOADS = {
    "short_qa": WorkloadSpec("short_qa", long_context_ratio=0.0, session_ratio=0.0, shared_prefix_ratio=0.1, utility_skew=1.0),
    "long_context": WorkloadSpec("long_context", long_context_ratio=1.0, session_ratio=0.0, shared_prefix_ratio=0.4, utility_skew=1.0),
    "mixed_short_long": WorkloadSpec("mixed_short_long", long_context_ratio=0.45, session_ratio=0.15, shared_prefix_ratio=0.3, utility_skew=1.5),
    "agent_session": WorkloadSpec("agent_session", long_context_ratio=0.35, session_ratio=0.85, shared_prefix_ratio=0.5, utility_skew=2.0),
    "shared_system_prompt": WorkloadSpec("shared_system_prompt", long_context_ratio=0.2, session_ratio=0.2, shared_prefix_ratio=0.9, utility_skew=1.0),
    "repeated_rag_context": WorkloadSpec("repeated_rag_context", long_context_ratio=0.9, session_ratio=0.2, shared_prefix_ratio=0.85, utility_skew=1.3),
    "agent_session_prefix": WorkloadSpec("agent_session_prefix", long_context_ratio=0.55, session_ratio=0.9, shared_prefix_ratio=0.75, utility_skew=2.0),
    "non_reuse_control": WorkloadSpec("non_reuse_control", long_context_ratio=0.6, session_ratio=0.1, shared_prefix_ratio=0.0, utility_skew=1.0),
}


def get_workload(name: str) -> WorkloadSpec:
    if name not in WORKLOADS:
        raise KeyError(f"unknown workload: {name}")
    return WORKLOADS[name]

