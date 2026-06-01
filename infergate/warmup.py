from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

from .cache_registry import PrefixEntry
from .schemas import LoadSnapshot


@dataclass
class WarmupStats:
    warmup_requests: int = 0
    warmup_tokens: int = 0
    warmup_latency_ms: list[float] = field(default_factory=list)
    warmup_cache_hit_contribution: int = 0


class WarmupManager:
    def __init__(
        self,
        model_id: str | None = None,
        budget_fraction: float = 0.10,
        cooldown_s: float = 60.0,
    ) -> None:
        self.model_id = model_id
        self.budget_fraction = budget_fraction
        self.cooldown_s = cooldown_s
        self.stats = WarmupStats()

    def budget_available(self, warmup_token_budget_used: int, total_prompt_tokens: int) -> bool:
        if total_prompt_tokens <= 0:
            return True
        return warmup_token_budget_used < self.budget_fraction * total_prompt_tokens

    def should_warmup(
        self,
        entry: PrefixEntry,
        load: LoadSnapshot,
        warmup_token_budget_used: int,
        total_prompt_tokens: int,
    ) -> bool:
        return (
            load.metrics_available
            and load.num_requests_waiting == 0
            and load.kv_cache_usage_perc < 0.65
            and self.budget_available(warmup_token_budget_used, total_prompt_tokens)
            and entry.predicted_reuse_count >= 2
            and bool(entry.prompt_text)
            and (
                entry.last_warmup_ts is None
                or time.time() - entry.last_warmup_ts >= self.cooldown_s
            )
        )

    def build_warmup_request(self, entry: PrefixEntry) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "messages": [{"role": "user", "content": entry.prompt_text}],
            "max_tokens": 1,
            "temperature": 0,
            "metadata": {
                "infergate_warmup": True,
                "x-cache-key": entry.cache_key,
            },
        }

    async def run_warmup(self, entry: PrefixEntry, forwarder: Any) -> bool:
        body = self.build_warmup_request(entry)
        started = time.perf_counter()
        try:
            await forwarder(copy.deepcopy(body), is_warmup=True)
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.stats.warmup_requests += 1
            self.stats.warmup_tokens += 1
            self.stats.warmup_latency_ms.append(elapsed_ms)
            self.stats.warmup_cache_hit_contribution += 1
            return True
        except Exception:
            return False
