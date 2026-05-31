from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from infergate.metrics import percentile

RUN_RE = re.compile(
    r"policy=(?P<policy>.+?)_workload=(?P<workload>.+?)_concurrency=(?P<concurrency>\d+)"
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
        "policy": groups["policy"],
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


def session_rates(trace_rows: list[dict[str, Any]]) -> tuple[float, float]:
    sessions: dict[str, list[dict[str, Any]]] = {}
    for row in trace_rows:
        sessions.setdefault(str(row.get("session_id", "default")), []).append(row)
    if not sessions:
        return 0.0, 0.0
    completed = 0
    full_quality = 0
    for rows in sessions.values():
        accepted = all(bool(row.get("accepted")) for row in rows)
        if accepted:
            completed += 1
        if accepted and all(row.get("decision") != "degrade" for row in rows):
            full_quality += 1
    return completed / len(sessions), full_quality / len(sessions)


def summarize_run(client_path: Path, trace_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_id = client_path.stem.removeprefix("client_")
    meta = parse_run_id(run_id)
    client_rows = read_jsonl(client_path)
    trace_rows = read_jsonl(trace_path)
    total = max(1, len(trace_rows))
    decisions: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for row in trace_rows:
        decision = str(row.get("decision", "unknown"))
        reason = str(row.get("reason", "unknown"))
        decisions[decision] = decisions.get(decision, 0) + 1
        reasons[reason] = reasons.get(reason, 0) + 1

    accepted_rows = [row for row in trace_rows if row.get("accepted")]
    accepted = len(accepted_rows)
    non_2xx = sum(1 for row in client_rows if not row.get("accepted"))
    ttft = [safe_float(row.get("ttft_ms")) for row in accepted_rows if row.get("ttft_ms") is not None]
    e2e = [safe_float(row.get("e2e_ms")) for row in accepted_rows if row.get("e2e_ms") is not None]
    gateway = [safe_float(row.get("gateway_ms")) for row in trace_rows]
    runtime_s = max([safe_float(row.get("latency_ms")) for row in client_rows] or [0.0]) / 1000.0
    utility_goodput = sum(safe_float(row.get("utility"), 1.0) for row in accepted_rows)

    slo_met = 0
    for row in trace_rows:
        deadline = row.get("deadline_ms")
        if not row.get("accepted"):
            continue
        if deadline is None or deadline == "":
            slo_met += 1
        elif safe_float(row.get("e2e_ms")) <= safe_float(deadline):
            slo_met += 1

    step0_rows = [row for row in trace_rows if int(row.get("session_step") or 0) == 0]
    step0_rejects = [row for row in step0_rows if row.get("rejected")]
    degraded_rows = [row for row in trace_rows if row.get("decision") == "degrade"]
    degraded_correct = [
        row
        for row in degraded_rows
        if row.get("max_tokens_sent") is not None
        and row.get("max_tokens_original") is not None
        and safe_float(row["max_tokens_sent"]) < safe_float(row["max_tokens_original"])
    ]
    session_completion, full_quality_session_completion = session_rates(trace_rows)

    summary = {
        **meta,
        "client_path": str(client_path),
        "trace_path": str(trace_path),
        "trace_rows": len(trace_rows),
        "accepted": accepted,
        "non_2xx": non_2xx,
        "utility_weighted_goodput": utility_goodput,
        "utility_goodput_per_second": utility_goodput / max(runtime_s, 1e-9),
        "SLO_satisfaction_rate": slo_met / total,
        "session_completion_rate": session_completion,
        "full_quality_session_completion_rate": full_quality_session_completion,
        "TTFT_P50": percentile(ttft, 50),
        "TTFT_P95": percentile(ttft, 95),
        "TTFT_P99": percentile(ttft, 99),
        "E2E_P50": percentile(e2e, 50),
        "E2E_P95": percentile(e2e, 95),
        "E2E_P99": percentile(e2e, 99),
        "gateway_P95": percentile(gateway, 95),
        "gateway_max": max(gateway) if gateway else 0.0,
        "decision_accept": decisions.get("accept", 0),
        "decision_defer": decisions.get("defer", 0),
        "decision_degrade": decisions.get("degrade", 0),
        "decision_reject": decisions.get("reject", 0),
        "entry_rejection_rate": len(step0_rejects) / max(1, len(step0_rows)),
        "degraded_rate": decisions.get("degrade", 0) / total,
        "non_2xx_rate": non_2xx / max(1, len(client_rows)),
        "tokenizer_fallback_rate": sum(1 for row in trace_rows if row.get("tokenizer_fallback")) / total,
        "degrade_correct_rate": len(degraded_correct) / max(1, len(degraded_rows)),
        "runtime_s": runtime_s,
        "reason_counts": json.dumps(reasons, ensure_ascii=False, sort_keys=True),
    }
    breakdown = [
        {
            "run_id": run_id,
            "policy": meta.get("policy"),
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
    parser.add_argument("--input-dir", default="results/main_pilot")
    parser.add_argument("--output", default="results/main_pilot/summary.csv")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    summaries: list[dict[str, Any]] = []
    breakdown: list[dict[str, Any]] = []
    for client_path in sorted(input_dir.glob("client_*.jsonl")):
        run_id = client_path.stem.removeprefix("client_")
        trace_path = input_dir / f"trace_{run_id}.jsonl"
        summary, rows = summarize_run(client_path, trace_path)
        summaries.append(summary)
        breakdown.extend(rows)
    output = Path(args.output)
    write_csv(output, summaries)
    write_csv(output.parent / "decision_breakdown.csv", breakdown)
    print(f"wrote {output} rows={len(summaries)}")
    print(f"wrote {output.parent / 'decision_breakdown.csv'} rows={len(breakdown)}")


if __name__ == "__main__":
    main()

