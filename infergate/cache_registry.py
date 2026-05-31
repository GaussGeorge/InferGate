from __future__ import annotations

import time
from dataclasses import dataclass

from .schemas import CacheState


@dataclass
class PrefixEntry:
    cache_key: str
    prompt_text: str = ""
    seen_count: int = 0
    prefix_hit_count: int = 0
    utility_sum: float = 0.0
    total_prompt_tokens: int = 0
    last_seen_ts: float | None = None
    last_hit_ts: float | None = None
    warmed_count: int = 0

    @property
    def predicted_reuse_count(self) -> int:
        return max(0, self.seen_count - 1)


class CacheRegistry:
    def __init__(self, cache_backend: str = "vllm_apc") -> None:
        self.cache_backend = cache_backend
        self._entries: dict[str, PrefixEntry] = {}
        self.total_prompt_tokens = 0
        self.warmup_token_budget_used = 0

    def get_entry(self, cache_key: str) -> PrefixEntry:
        if cache_key not in self._entries:
            self._entries[cache_key] = PrefixEntry(cache_key=cache_key)
        return self._entries[cache_key]

    def observe(
        self,
        cache_key: str,
        prompt_tokens: int,
        utility: float,
        prompt_text: str = "",
        cache_hit: bool = False,
    ) -> PrefixEntry:
        entry = self.get_entry(cache_key)
        entry.seen_count += 1
        entry.utility_sum += utility
        entry.total_prompt_tokens += max(0, prompt_tokens)
        entry.last_seen_ts = time.time()
        if prompt_text and not entry.prompt_text:
            entry.prompt_text = prompt_text
        if cache_hit:
            entry.prefix_hit_count += 1
            entry.last_hit_ts = entry.last_seen_ts
        self.total_prompt_tokens += max(0, prompt_tokens)
        return entry

    def mark_warmup(self, cache_key: str, warmup_tokens: int = 1) -> None:
        entry = self.get_entry(cache_key)
        entry.warmed_count += 1
        self.warmup_token_budget_used += max(0, warmup_tokens)

    def state(self, cache_key: str) -> CacheState:
        entry = self.get_entry(cache_key)
        return CacheState(
            cache_key=cache_key,
            cache_backend=self.cache_backend,
            seen_count=entry.seen_count,
            prefix_hit_count=entry.prefix_hit_count,
            predicted_reuse_count=entry.predicted_reuse_count,
            utility_sum=entry.utility_sum,
            total_prompt_tokens=entry.total_prompt_tokens,
            last_seen_ts=entry.last_seen_ts,
            last_hit_ts=entry.last_hit_ts,
        )

    def candidates(self, min_reuse_count: int = 2) -> list[PrefixEntry]:
        entries = [
            entry for entry in self._entries.values() if entry.predicted_reuse_count >= min_reuse_count
        ]
        return sorted(entries, key=lambda item: (item.utility_sum, item.predicted_reuse_count), reverse=True)

