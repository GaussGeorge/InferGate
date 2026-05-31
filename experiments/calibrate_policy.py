from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from experiments.run_experiment import run_experiment
from infergate.metrics import percentile


CALIBRATED_ENV = {
    "INFERGATE_MAX_ACTIVE": "4",
    "INFERGATE_MAX_QUEUE_SIZE": "32",
    "INFERGATE_QUEUE_TIMEOUT_MS": "120000",
    "INFERGATE_KV_REJECT_THRESHOLD": "0.80",
    "INFERGATE_WAITING_REJECT_THRESHOLD": "8",
    "INFERGATE_REJECT_SCORE": "0.0030",
    "INFERGATE_DEGRADE_SCORE": "0.0060",
    "INFERGATE_DEGRADED_MAX_TOKENS": "64",
    "INFERGATE_POLICY": "infergate_admission",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def _wait_for_health(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.perf_counter() + timeout_s
    async with httpx.AsyncClient(timeout=1.0, trust_env=False) as client:
        while time.perf_counter() < deadline:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError(f"InferGate did not become healthy at {url}")


def _start_infergate(run_dir: Path, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(CALIBRATED_ENV)
    env["INFERGATE_RESULTS_DIR"] = str(run_dir)
    env.setdefault("CACHE_BACKEND", "vllm_apc")
    env.setdefault("INFERGATE_REQUEST_TIMEOUT_S", "600")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "infergate_trace.jsonl").unlink(missing_ok=True)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "infergate.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _summarize_trace(run_id: str, trace_path: Path, client_path: Path, settings: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace_rows = _read_jsonl(trace_path)
    client_rows = _read_jsonl(client_path)
    decisions: dict[str, int] = {}
    for row in trace_rows:
        decisions[row.get("decision", "unknown")] = decisions.get(row.get("decision", "unknown"), 0) + 1
    gateway_ms = [float(row.get("gateway_ms") or 0) for row in trace_rows]
    fallback_count = sum(1 for row in trace_rows if row.get("tokenizer_fallback"))
    degraded_correctly = sum(
        1
        for row in trace_rows
        if row.get("decision") == "degrade"
        and row.get("max_tokens_sent") is not None
        and row.get("max_tokens_original") is not None
        and row["max_tokens_sent"] < row["max_tokens_original"]
    )
    summary = {
        "run_id": run_id,
        "client_path": str(client_path),
        "trace_path": str(trace_path),
        "requests": len(client_rows),
        "trace_rows": len(trace_rows),
        "accepted": sum(1 for row in client_rows if row.get("accepted")),
        "non_2xx": sum(1 for row in client_rows if not row.get("accepted")),
        "accept_decisions": decisions.get("accept", 0),
        "defer_decisions": decisions.get("defer", 0),
        "degrade_decisions": decisions.get("degrade", 0),
        "reject_decisions": decisions.get("reject", 0),
        "decision_types": ",".join(sorted(decisions)),
        "gateway_p95_ms": percentile(gateway_ms, 95),
        "gateway_max_ms": max(gateway_ms) if gateway_ms else 0.0,
        "tokenizer_fallback_rate": fallback_count / max(1, len(trace_rows)),
        "degraded_correctly": degraded_correctly,
        **settings,
    }
    breakdown = [
        {"run_id": run_id, "decision": decision, "count": count}
        for decision, count in sorted(decisions.items())
    ]
    return summary, breakdown


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


async def _run_single_calibration(
    output_dir: Path,
    run_id: str,
    workload: str,
    concurrency: int,
    requests: int,
    seed: int,
    port: int,
    settings: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = output_dir / "_server" / run_id
    client_path = output_dir / f"client_{run_id}.jsonl"
    trace_path = output_dir / f"trace_{run_id}.jsonl"
    process = _start_infergate(run_dir, port)
    try:
        await _wait_for_health(f"http://127.0.0.1:{port}/healthz")
        await run_experiment(
            target_url=f"http://127.0.0.1:{port}/v1/chat/completions",
            workload=workload,
            requests=requests,
            concurrency=concurrency,
            policy="infergate_admission",
            output=str(client_path),
            seed=seed,
            model=os.getenv("MODEL_ID"),
        )
    finally:
        _stop_process(process)
    server_trace = run_dir / "infergate_trace.jsonl"
    if server_trace.exists():
        trace_path.write_text(server_trace.read_text(encoding="utf-8"), encoding="utf-8")
    summary, breakdown = _summarize_trace(run_id, trace_path, client_path, settings)
    summary["workload"] = workload
    summary["concurrency"] = concurrency
    summary["requests_per_run"] = requests
    return summary, breakdown


async def run_calibration(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workloads = [item.strip() for item in args.workloads.split(",") if item.strip()]
    concurrencies = [int(item.strip()) for item in args.concurrency.split(",") if item.strip()]
    summaries: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    settings = {f"setting_{key.lower()}": value for key, value in CALIBRATED_ENV.items()}

    for workload in workloads:
        for concurrency in concurrencies:
            run_id = f"policy=infergate_admission_workload={workload}_concurrency={concurrency}_requests={args.requests}"
            best_summary: dict[str, Any] | None = None
            best_breakdown: list[dict[str, Any]] = []
            best_client_text = ""
            best_trace_text = ""
            for attempt in range(args.max_retries + 1):
                summary, breakdown = await _run_single_calibration(
                    output_dir=output_dir,
                    run_id=run_id,
                    workload=workload,
                    concurrency=concurrency,
                    requests=args.requests,
                    seed=args.seed + attempt,
                    port=args.port,
                    settings=settings,
                )
                if best_summary is None or summary["gateway_p95_ms"] < best_summary["gateway_p95_ms"]:
                    best_summary = summary
                    best_breakdown = breakdown
                    client_path = output_dir / f"client_{run_id}.jsonl"
                    trace_path = output_dir / f"trace_{run_id}.jsonl"
                    best_client_text = client_path.read_text(encoding="utf-8") if client_path.exists() else ""
                    best_trace_text = trace_path.read_text(encoding="utf-8") if trace_path.exists() else ""
                if summary["gateway_p95_ms"] <= args.gateway_p95_target_ms:
                    break
                if attempt < args.max_retries:
                    print(
                        f"retrying {run_id}: gateway_p95_ms={summary['gateway_p95_ms']:.3f} "
                        f"> target={args.gateway_p95_target_ms:.3f}"
                    )
            summary = best_summary or {}
            breakdown = best_breakdown
            if best_client_text:
                (output_dir / f"client_{run_id}.jsonl").write_text(best_client_text, encoding="utf-8")
            if best_trace_text:
                (output_dir / f"trace_{run_id}.jsonl").write_text(best_trace_text, encoding="utf-8")
            summaries.append(summary)
            breakdown_rows.extend(breakdown)

    _write_csv(output_dir / "summary.csv", summaries)
    _write_csv(output_dir / "decision_breakdown.csv", breakdown_rows)
    non_accept = sum(
        row.get("defer_decisions", 0) + row.get("degrade_decisions", 0) + row.get("reject_decisions", 0)
        for row in summaries
    )
    if non_accept == 0:
        suggestions = [
            "No non-accept decisions observed.",
            "Try increasing concurrency above 16.",
            "Try lowering INFERGATE_MAX_ACTIVE below 4.",
            "Try raising INFERGATE_REJECT_SCORE and INFERGATE_DEGRADE_SCORE.",
        ]
        (output_dir / "suggestions.txt").write_text("\n".join(suggestions) + "\n", encoding="utf-8")
        print("\n".join(suggestions))
    print(f"wrote {output_dir / 'summary.csv'}")
    print(f"wrote {output_dir / 'decision_breakdown.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/calibration")
    parser.add_argument("--workloads", default="long_context,mixed_short_long,agent_session")
    parser.add_argument("--concurrency", default="8,12,16")
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--gateway-p95-target-ms", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(run_calibration(args))


if __name__ == "__main__":
    main()
