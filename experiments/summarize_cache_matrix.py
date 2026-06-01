from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from infergate.metrics import load_snapshot_from_prometheus, percentile

RUN_RE = re.compile(
    r"cache_mode=(?P<cache_mode>.+?)_workload=(?P<workload>.+?)_concurrency=(?P<concurrency>\d+)"
    r"_repeat=(?P<repeat>\d+)_requests=(?P<requests>\d+)"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_run_id(run_id: str) -> dict[str, Any]:
    match = RUN_RE.fullmatch(run_id)
    if not match:
        return {"run_id": run_id}
    groups = match.groupdict()
    return {
        "run_id": run_id,
        "cache_mode": groups["cache_mode"],
        "workload": groups["workload"],
        "concurrency": int(groups["concurrency"]),
        "repeat": int(groups["repeat"]),
        "requests": int(groups["requests"]),
    }


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _snapshot(path: Path):
    if not path.exists():
        return load_snapshot_from_prometheus("")
    return load_snapshot_from_prometheus(path.read_text(encoding="utf-8"))


def summarize_run(client_path: Path, trace_path: Path, metrics_before_path: Path, metrics_after_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_id = client_path.stem.removeprefix("client_")
    meta = parse_run_id(run_id)
    client_rows = read_jsonl(client_path)
    trace_rows = read_jsonl(trace_path)
    before = _snapshot(metrics_before_path)
    after = _snapshot(metrics_after_path)
    total = max(1, len(trace_rows))
    accepted_rows = [row for row in trace_rows if row.get("accepted")]
    decisions: dict[str, int] = {}
    for row in trace_rows:
        decision = str(row.get("decision", "unknown"))
        decisions[decision] = decisions.get(decision, 0) + 1

    ttft = [safe_float(row.get("ttft_ms")) for row in accepted_rows if row.get("ttft_ms") is not None]
    e2e = [safe_float(row.get("e2e_ms")) for row in accepted_rows if row.get("e2e_ms") is not None]
    runtime_s = max([safe_float(row.get("latency_ms")) for row in client_rows] or [0.0]) / 1000.0
    utility_goodput = sum(safe_float(row.get("utility"), 1.0) for row in accepted_rows)
    prompt_tokens = sum(max(0.0, safe_float(row.get("prompt_tokens"))) for row in trace_rows)
    warmup_tokens = max([safe_float(row.get("warmup_tokens_used")) for row in trace_rows] or [0.0])
    queries_before = before.prefix_cache_queries_total or 0.0
    queries_after = after.prefix_cache_queries_total or 0.0
    hits_before = before.prefix_cache_hits_total or 0.0
    hits_after = after.prefix_cache_hits_total or 0.0
    query_delta = queries_after - queries_before
    hit_delta = hits_after - hits_before
    before_rate = before.prefix_cache_hit_rate
    after_rate = after.prefix_cache_hit_rate
    rate_delta = None
    if before_rate is not None and after_rate is not None:
        rate_delta = after_rate - before_rate
    elif query_delta > 0:
        rate_delta = hit_delta / query_delta

    step0_rows = [row for row in trace_rows if int(row.get("session_step") or 0) == 0]
    step0_rejects = [row for row in step0_rows if row.get("rejected")]
    summary = {
        **meta,
        "client_path": str(client_path),
        "trace_path": str(trace_path),
        "metrics_before_path": str(metrics_before_path),
        "metrics_after_path": str(metrics_after_path),
        "trace_rows": len(trace_rows),
        "accepted": len(accepted_rows),
        "non_2xx": sum(1 for row in client_rows if not row.get("accepted")),
        "prefix_cache_queries_before": queries_before,
        "prefix_cache_queries_after": queries_after,
        "prefix_cache_queries_delta": query_delta,
        "prefix_cache_hits_before": hits_before,
        "prefix_cache_hits_after": hits_after,
        "prefix_cache_hits_delta": hit_delta,
        "prefix_cache_hit_rate_before": before_rate,
        "prefix_cache_hit_rate_after": after_rate,
        "prefix_cache_hit_rate_delta": rate_delta,
        "kv_cache_usage_before": before.kv_cache_usage_perc,
        "kv_cache_usage_after": after.kv_cache_usage_perc,
        "TTFT_P50": percentile(ttft, 50),
        "TTFT_P95": percentile(ttft, 95),
        "TTFT_P99": percentile(ttft, 99),
        "E2E_P95": percentile(e2e, 95),
        "utility_goodput_per_second": utility_goodput / max(runtime_s, 1e-9),
        "warmup_token_overhead_ratio": warmup_tokens / max(prompt_tokens, 1.0),
        "warmup_requests": sum(1 for row in trace_rows if row.get("warmup_sent")),
        "warmup_candidates": sum(1 for row in trace_rows if row.get("warmup_candidate")),
        "degraded_rate": decisions.get("degrade", 0) / total,
        "entry_rejection_rate": len(step0_rejects) / max(1, len(step0_rows)),
        "decision_accept": decisions.get("accept", 0),
        "decision_defer": decisions.get("defer", 0),
        "decision_degrade": decisions.get("degrade", 0),
        "decision_reject": decisions.get("reject", 0),
    }
    breakdown = [
        {
            "run_id": run_id,
            "cache_mode": meta.get("cache_mode"),
            "workload": meta.get("workload"),
            "concurrency": meta.get("concurrency"),
            "repeat": meta.get("repeat"),
            "decision": decision,
            "count": count,
        }
        for decision, count in sorted(decisions.items())
    ]
    return summary, breakdown


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/cache_matrix")
    parser.add_argument("--output", default="results/cache_matrix/summary.csv")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    summaries: list[dict[str, Any]] = []
    breakdown: list[dict[str, Any]] = []
    for client_path in sorted(input_dir.glob("client_*.jsonl")):
        run_id = client_path.stem.removeprefix("client_")
        summary, rows = summarize_run(
            client_path,
            input_dir / f"trace_{run_id}.jsonl",
            input_dir / f"metrics_before_{run_id}.txt",
            input_dir / f"metrics_after_{run_id}.txt",
        )
        summaries.append(summary)
        breakdown.extend(rows)
    output = Path(args.output)
    write_csv(output, summaries)
    write_csv(output.parent / "decision_breakdown.csv", breakdown)
    print(f"wrote {output} rows={len(summaries)}")
    print(f"wrote {output.parent / 'decision_breakdown.csv'} rows={len(breakdown)}")


if __name__ == "__main__":
    main()
