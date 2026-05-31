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

