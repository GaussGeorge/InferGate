from pathlib import Path

from infergate.metrics import load_snapshot_from_prometheus, parse_prometheus_metrics


def test_parse_prometheus_metrics_with_labels() -> None:
    text = """
# HELP vllm:num_requests_waiting waiting
vllm:num_requests_waiting{model_name="qwen"} 3
vllm:num_requests_running 2
vllm:gpu_cache_usage_perc 0.61
"""
    samples = parse_prometheus_metrics(text)
    assert len(samples) == 3
    assert samples[0].labels["model_name"] == "qwen"
    snapshot = load_snapshot_from_prometheus(text)
    assert snapshot.num_requests_waiting == 3
    assert snapshot.num_requests_running == 2
    assert snapshot.kv_cache_usage_perc == 0.61
    assert snapshot.metrics_available is True


def test_parse_percent_cache_usage() -> None:
    snapshot = load_snapshot_from_prometheus("vllm:gpu_cache_usage_perc 73\n")
    assert snapshot.kv_cache_usage_perc == 0.73


def test_parse_real_vllm_a4000_metrics_sample() -> None:
    sample_path = Path(__file__).parent / "fixtures" / "real_vllm_metrics_sample.txt"
    snapshot = load_snapshot_from_prometheus(sample_path.read_text(encoding="utf-8"))
    assert snapshot.metrics_available is True
    assert snapshot.num_requests_running == 0
    assert snapshot.num_requests_waiting == 0
    assert snapshot.kv_cache_usage_perc == 0.0
    assert snapshot.prefix_cache_queries_total is not None
    assert snapshot.prefix_cache_hits_total is not None
    assert snapshot.prefix_cache_hit_rate is None or 0.0 <= snapshot.prefix_cache_hit_rate <= 1.0
    dumped = snapshot.model_dump()
    assert dumped["is_overloaded"] is False
