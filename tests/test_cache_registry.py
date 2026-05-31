from infergate.cache_registry import CacheRegistry
from infergate.schemas import LoadSnapshot
from infergate.warmup import WarmupManager


def test_cache_registry_tracks_reuse_and_hits() -> None:
    registry = CacheRegistry(cache_backend="vllm_apc")
    registry.observe("k1", prompt_tokens=100, utility=2.0, prompt_text="prefix")
    registry.observe("k1", prompt_tokens=120, utility=1.5, cache_hit=True)
    state = registry.state("k1")
    assert state.seen_count == 2
    assert state.predicted_reuse_count == 1
    assert state.prefix_hit_count == 1
    assert state.utility_sum == 3.5
    assert registry.total_prompt_tokens == 220


def test_warmup_budget_conditions() -> None:
    registry = CacheRegistry()
    for _ in range(3):
        registry.observe("shared", prompt_tokens=100, utility=1.0, prompt_text="shared prefix")
    entry = registry.get_entry("shared")
    manager = WarmupManager(model_id="mock", budget_fraction=0.10)
    low_load = LoadSnapshot(num_requests_waiting=0, kv_cache_usage_perc=0.5, metrics_available=True)
    high_load = LoadSnapshot(num_requests_waiting=1, kv_cache_usage_perc=0.5, metrics_available=True)
    assert manager.should_warmup(entry, low_load, 0, registry.total_prompt_tokens) is True
    assert manager.should_warmup(entry, high_load, 0, registry.total_prompt_tokens) is False
    assert manager.should_warmup(entry, low_load, 31, registry.total_prompt_tokens) is False

