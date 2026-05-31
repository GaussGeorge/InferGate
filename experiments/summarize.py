from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from infergate.metrics import percentile


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_file(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    latencies = [float(row.get("latency_ms") or row.get("e2e_ms") or 0) for row in rows if row.get("accepted")]
    accepted = sum(1 for row in rows if row.get("accepted"))
    rejected = sum(1 for row in rows if row.get("rejected"))
    degraded = sum(1 for row in rows if row.get("degraded"))
    total_prompt = sum(int(row.get("prompt_tokens") or 0) for row in rows)
    total_completion = sum(int(row.get("completion_tokens") or 0) for row in rows)
    utility_goodput = sum(float(row.get("utility") or 1.0) for row in rows if row.get("accepted"))
    return {
        "path": str(path),
        "policy": rows[0].get("policy") if rows else "unknown",
        "requests": len(rows),
        "accepted": accepted,
        "rejected": rejected,
        "degraded": degraded,
        "utility_weighted_goodput": utility_goodput,
        "slo_satisfaction_rate": accepted / max(1, len(rows)),
        "entry_rejection_rate": rejected / max(1, len(rows)),
        "wasted_token_ratio": total_completion / max(1, total_prompt + total_completion),
        "e2e_p50_ms": percentile(latencies, 50),
        "e2e_p95_ms": percentile(latencies, 95),
        "e2e_p99_ms": percentile(latencies, 99),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results")
    parser.add_argument("--output", default="results/summary.csv")
    args = parser.parse_args()
    paths = sorted(Path(args.input_dir).rglob("*.jsonl"))
    rows = [summarize_file(path) for path in paths if path.name != "infergate_trace.jsonl"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"wrote {output} rows={len(df)}")


if __name__ == "__main__":
    main()

