from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from experiments.run_experiment import _commit_hash, _runtime_metadata, run_experiment


POLICIES = ["fcfs", "sjf", "edf", "static_threshold", "vtc_inspired", "infergate_admission"]
PILOT_WORKLOADS = ["long_context", "mixed_short_long", "agent_session"]
MAIN_WORKLOADS = ["short_qa", "long_context", "mixed_short_long", "agent_session"]
CONCURRENCY = [8, 12, 16]

CALIBRATED_ENV = {
    "INFERGATE_MAX_ACTIVE": "4",
    "INFERGATE_MAX_QUEUE_SIZE": "32",
    "INFERGATE_QUEUE_TIMEOUT_MS": "120000",
    "INFERGATE_KV_REJECT_THRESHOLD": "0.80",
    "INFERGATE_WAITING_REJECT_THRESHOLD": "8",
    "INFERGATE_REJECT_SCORE": "0.0030",
    "INFERGATE_DEGRADE_SCORE": "0.0060",
    "INFERGATE_DEGRADED_MAX_TOKENS": "64",
    "CACHE_BACKEND": "vllm_apc",
    "INFERGATE_REQUEST_TIMEOUT_S": "600",
}


def run_id(policy: str, workload: str, concurrency: int, repeat: int, requests: int) -> str:
    return (
        f"policy={policy}_workload={workload}_concurrency={concurrency}"
        f"_repeat={repeat}_requests={requests}"
    )


async def wait_for_health(url: str, timeout_s: float = 120.0) -> None:
    deadline = time.perf_counter() + timeout_s
    async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
        while time.perf_counter() < deadline:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError(f"InferGate did not become healthy at {url}")


def start_infergate(run_dir: Path, port: int, default_policy: str) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(CALIBRATED_ENV)
    env["INFERGATE_POLICY"] = default_policy
    env["INFERGATE_RESULTS_DIR"] = str(run_dir)
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


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def write_run_manifest(output_dir: Path, record: dict[str, Any]) -> None:
    manifest_path = output_dir / "matrix_manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_one(
    output_dir: Path,
    policy: str,
    workload: str,
    concurrency: int,
    repeat: int,
    requests: int,
    seed: int,
    port: int,
    force: bool,
) -> None:
    rid = run_id(policy, workload, concurrency, repeat, requests)
    run_dir = output_dir / "_server" / rid
    client_path = output_dir / f"client_{rid}.jsonl"
    trace_path = output_dir / f"trace_{rid}.jsonl"
    if client_path.exists() and trace_path.exists() and not force:
        print(f"skip existing {rid}")
        return

    started = time.time()
    process = start_infergate(run_dir, port, policy)
    status = "completed"
    error: str | None = None
    summary: dict[str, Any] | None = None
    try:
        await wait_for_health(f"http://127.0.0.1:{port}/healthz")
        summary = await run_experiment(
            target_url=f"http://127.0.0.1:{port}/v1/chat/completions",
            workload=workload,
            requests=requests,
            concurrency=concurrency,
            policy=policy,
            output=str(client_path),
            seed=seed,
            model=os.getenv("MODEL_ID"),
        )
    except Exception as exc:
        status = "failed"
        error = str(exc)
        raise
    finally:
        stop_process(process)
        server_trace = run_dir / "infergate_trace.jsonl"
        if server_trace.exists():
            trace_path.write_text(server_trace.read_text(encoding="utf-8"), encoding="utf-8")
        write_run_manifest(
            output_dir,
            {
                "ts": started,
                "status": status,
                "error": error,
                "run_id": rid,
                "policy": policy,
                "workload": workload,
                "concurrency": concurrency,
                "repeat": repeat,
                "requests": requests,
                "seed": seed,
                "client_path": str(client_path),
                "trace_path": str(trace_path),
                "commit": _commit_hash(),
                "model": os.getenv("MODEL_ID"),
                "tokenizer_id": os.getenv("INFERGATE_TOKENIZER_ID"),
                "vllm_base_url": os.getenv("VLLM_BASE_URL"),
                "vllm_metrics_url": os.getenv("VLLM_METRICS_URL"),
                **_runtime_metadata(os.getenv("MODEL_ID")),
                "settings": CALIBRATED_ENV,
                "summary": summary,
            },
        )


async def run_matrix(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "pilot":
        default_requests = 20
        default_repeats = 1
        default_workloads = PILOT_WORKLOADS
    else:
        default_requests = 60
        default_repeats = 3
        default_workloads = MAIN_WORKLOADS

    policies = [item.strip() for item in (args.policies or ",".join(POLICIES)).split(",") if item.strip()]
    workloads = [item.strip() for item in (args.workloads or ",".join(default_workloads)).split(",") if item.strip()]
    concurrencies = [int(item.strip()) for item in (args.concurrency or ",".join(map(str, CONCURRENCY))).split(",") if item.strip()]
    requests = args.requests or default_requests
    repeats = args.repeats or default_repeats

    for repeat in range(repeats):
        seed = args.seed + repeat
        for policy in policies:
            for workload in workloads:
                for concurrency in concurrencies:
                    await run_one(output_dir, policy, workload, concurrency, repeat, requests, seed, args.port, args.force)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "main"], default="pilot")
    parser.add_argument("--output-dir", default="results/main_pilot")
    parser.add_argument("--policies", default=None)
    parser.add_argument("--workloads", default=None)
    parser.add_argument("--concurrency", default=None)
    parser.add_argument("--requests", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_matrix(args))


if __name__ == "__main__":
    main()
