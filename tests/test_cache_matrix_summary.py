import json

from experiments.summarize_cache_matrix import summarize_run


METRICS_BEFORE = """
vllm:kv_cache_usage_perc 0.2
vllm:prefix_cache_queries_total{model_name="qwen"} 10
vllm:prefix_cache_hits_total{model_name="qwen"} 2
"""

METRICS_AFTER = """
vllm:kv_cache_usage_perc 0.3
vllm:prefix_cache_queries_total{model_name="qwen"} 20
vllm:prefix_cache_hits_total{model_name="qwen"} 8
"""


def test_summarize_cache_matrix_metrics(tmp_path) -> None:
    run_id = "cache_mode=infergate_cache_workload=repeated_rag_context_concurrency=4_repeat=0_requests=2"
    client_path = tmp_path / f"client_{run_id}.jsonl"
    trace_path = tmp_path / f"trace_{run_id}.jsonl"
    before_path = tmp_path / f"metrics_before_{run_id}.txt"
    after_path = tmp_path / f"metrics_after_{run_id}.txt"
    client_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"accepted": True, "latency_ms": 1000, "utility": 2.0},
                {"accepted": True, "latency_ms": 1200, "utility": 1.0},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trace_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "accepted": True,
                    "decision": "accept",
                    "utility": 2.0,
                    "ttft_ms": 100,
                    "e2e_ms": 900,
                    "prompt_tokens": 100,
                    "session_step": 0,
                    "warmup_candidate": True,
                    "warmup_sent": True,
                    "warmup_tokens_used": 1,
                },
                {
                    "accepted": True,
                    "decision": "defer",
                    "utility": 1.0,
                    "ttft_ms": 120,
                    "e2e_ms": 1100,
                    "prompt_tokens": 120,
                    "session_step": 0,
                    "warmup_candidate": False,
                    "warmup_sent": False,
                    "warmup_tokens_used": 1,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    before_path.write_text(METRICS_BEFORE, encoding="utf-8")
    after_path.write_text(METRICS_AFTER, encoding="utf-8")
    summary, breakdown = summarize_run(client_path, trace_path, before_path, after_path)
    assert summary["cache_mode"] == "infergate_cache"
    assert summary["prefix_cache_queries_delta"] == 10
    assert summary["prefix_cache_hits_delta"] == 6
    assert summary["warmup_requests"] == 1
    assert 0 < summary["warmup_token_overhead_ratio"] < 0.01
    assert {row["decision"] for row in breakdown} == {"accept", "defer"}
