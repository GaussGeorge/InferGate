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


CACHE_MODES = ["no_cache_control", "always_warm", "lru_warm", "infergate_cache"]
WORKLOADS = ["shared_system_prompt", "repeated_rag_context", "agent_session_prefix", "non_reuse_control"]
CONCURRENCY = [4, 8, 12]

CALIBRATED_ENV = {
    "INFERGATE_MAX_ACTIVE": "4",
    "INFERGATE_MAX_QUEUE_SIZE": "32",
    "INFERGATE_QUEUE_TIMEOUT_MS": "120000",
    "INFERGATE_KV_REJECT_THRESHOLD": "0.80",
    "INFERGATE_WAITING_REJECT_THRESHOLD": "8",
    "INFERGATE_REJECT_SCORE": "0.0030",
    "INFERGATE_DEGRADE_SCORE": "0.0060",
    "INFERGATE_DEGRADED_MAX_TOKENS": "64",
    "INFERGATE_WARMUP_BUDGET_FRACTION": "0.10",
    "INFERGATE_WARMUP_COOLDOWN_S": "30",
    "CACHE_BACKEND": "vllm_apc",
    "INFERGATE_REQUEST_TIMEOUT_S": "600",
}


def policy_for_mode(cache_mode: str) -> str:
    return "infergate_cache" if cache_mode == "infergate_cache" else "infergate_admission"


def run_id(cache_mode: str, workload: str, concurrency: int, repeat: int, requests: int) -> str:
    return (
        f"cache_mode={cache_mode}_workload={workload}_concurrency={concurrency}"
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


async def fetch_metrics_text(metrics_url: str) -> str:
    async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
        response = await client.get(metrics_url)
        response.raise_for_status()
        return response.text


def start_infergate(run_dir: Path, port: int, cache_mode: str) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(CALIBRATED_ENV)
    env["INFERGATE_POLICY"] = policy_for_mode(cache_mode)
    env["INFERGATE_CACHE_MODE"] = cache_mode
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
    manifest_path = output_dir / "cache_manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_one(
    output_dir: Path,
    cache_mode: str,
    workload: str,
    concurrency: int,
    repeat: int,
    requests: int,
    seed: int,
    port: int,
    force: bool,
) -> None:
    rid = run_id(cache_mode, workload, concurrency, repeat, requests)
    policy = policy_for_mode(cache_mode)
    run_dir = output_dir / "_server" / rid
    client_path = output_dir / f"client_{rid}.jsonl"
    trace_path = output_dir / f"trace_{rid}.jsonl"
    metrics_before_path = output_dir / f"metrics_before_{rid}.txt"
    metrics_after_path = output_dir / f"metrics_after_{rid}.txt"
    if client_path.exists() and trace_path.exists() and metrics_before_path.exists() and metrics_after_path.exists() and not force:
        print(f"skip existing {rid}")
        return

    started = time.time()
    process = start_infergate(run_dir, port, cache_mode)
    status = "completed"
    error: str | None = None
    summary: dict[str, Any] | None = None
    metrics_url = os.getenv("VLLM_METRICS_URL", "http://127.0.0.1:9999/metrics")
    try:
        await wait_for_health(f"http://127.0.0.1:{port}/healthz")
        metrics_before_path.write_text(await fetch_metrics_text(metrics_url), encoding="utf-8")
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
        metrics_after_path.write_text(await fetch_metrics_text(metrics_url), encoding="utf-8")
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
                "cache_mode": cache_mode,
                "policy": policy,
                "workload": workload,
                "concurrency": concurrency,
                "repeat": repeat,
                "requests": requests,
                "seed": seed,
                "client_path": str(client_path),
                "trace_path": str(trace_path),
                "metrics_before_path": str(metrics_before_path),
                "metrics_after_path": str(metrics_after_path),
                "commit": _commit_hash(),
                "model": os.getenv("MODEL_ID"),
                "tokenizer_id": os.getenv("INFERGATE_TOKENIZER_ID"),
                "vllm_base_url": os.getenv("VLLM_BASE_URL"),
                "vllm_metrics_url": metrics_url,
                **_runtime_metadata(os.getenv("MODEL_ID")),
                "settings": CALIBRATED_ENV | {"INFERGATE_CACHE_MODE": cache_mode},
                "summary": summary,
            },
        )


async def run_matrix(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_modes = [item.strip() for item in (args.cache_modes or ",".join(CACHE_MODES)).split(",") if item.strip()]
    workloads = [item.strip() for item in (args.workloads or ",".join(WORKLOADS)).split(",") if item.strip()]
    concurrencies = [int(item.strip()) for item in (args.concurrency or ",".join(map(str, CONCURRENCY))).split(",") if item.strip()]
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        for cache_mode in cache_modes:
            for workload in workloads:
                for concurrency in concurrencies:
                    await run_one(
                        output_dir,
                        cache_mode,
                        workload,
                        concurrency,
                        repeat,
                        args.requests,
                        seed,
                        args.port,
                        args.force,
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/cache_matrix")
    parser.add_argument("--cache-modes", default=None)
    parser.add_argument("--workloads", default=None)
    parser.add_argument("--concurrency", default=None)
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_matrix(args))


if __name__ == "__main__":
    main()
