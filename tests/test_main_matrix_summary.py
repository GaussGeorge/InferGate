import json

from experiments.summarize_main_matrix import summarize_run


def test_summarize_main_matrix_metrics(tmp_path) -> None:
    run_id = "policy=infergate_admission_workload=long_context_concurrency=8_repeat=0_requests=2"
    client_path = tmp_path / f"client_{run_id}.jsonl"
    trace_path = tmp_path / f"trace_{run_id}.jsonl"
    client_rows = [
        {"accepted": True, "latency_ms": 1000, "utility": 2.0, "policy": "infergate_admission"},
        {"accepted": False, "latency_ms": 200, "utility": 1.0, "policy": "infergate_admission"},
    ]
    trace_rows = [
        {
            "accepted": True,
            "rejected": False,
            "decision": "degrade",
            "reason": "degrade_low_score_overload",
            "utility": 2.0,
            "ttft_ms": 100,
            "e2e_ms": 900,
            "gateway_ms": 1,
            "deadline_ms": 1000,
            "session_id": "s1",
            "session_step": 0,
            "session_total_steps": 1,
            "tokenizer_fallback": False,
            "max_tokens_original": 128,
            "max_tokens_sent": 64,
        },
        {
            "accepted": False,
            "rejected": True,
            "decision": "reject",
            "reason": "low_score_overload",
            "utility": 1.0,
            "gateway_ms": 2,
            "deadline_ms": 1000,
            "session_id": "s2",
            "session_step": 0,
            "session_total_steps": 1,
            "tokenizer_fallback": False,
            "max_tokens_original": 256,
            "max_tokens_sent": None,
        },
    ]
    client_path.write_text("\n".join(json.dumps(row) for row in client_rows) + "\n", encoding="utf-8")
    trace_path.write_text("\n".join(json.dumps(row) for row in trace_rows) + "\n", encoding="utf-8")
    summary, breakdown = summarize_run(client_path, trace_path)
    assert summary["policy"] == "infergate_admission"
    assert summary["workload"] == "long_context"
    assert summary["concurrency"] == 8
    assert summary["repeat"] == 0
    assert summary["requests"] == 2
    assert summary["decision_degrade"] == 1
    assert summary["decision_reject"] == 1
    assert summary["degrade_correct_rate"] == 1.0
    assert summary["tokenizer_fallback_rate"] == 0.0
    assert {row["decision"] for row in breakdown} == {"degrade", "reject"}

