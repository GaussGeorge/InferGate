from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from infergate.metrics import percentile
from workloads.generator import generate_requests


def _commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "uncommitted"


def _update_manifest(output: Path, summary: dict[str, Any], config: dict[str, Any]) -> None:
    root = output.resolve()
    while root.parent != root and root.name != "results":
        root = root.parent
    manifest_path = (root if root.name == "results" else output.parent) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {"runs": []}
    else:
        manifest = {"runs": []}
    manifest.setdefault("runs", []).append(
        {
            "ts": time.time(),
            "commit": _commit_hash(),
            "output": str(output),
            "config": config,
            "summary": summary,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


async def _send_one(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    policy: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    headers = {
        "x-infergate-policy": policy,
    }
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    for key in [
        "x-session-id",
        "x-session-step",
        "x-session-total-steps",
        "x-request-utility",
        "x-request-deadline-ms",
        "x-cache-key",
    ]:
        if key in metadata:
            headers[key] = str(metadata[key])
    started = time.perf_counter()
    async with semaphore:
        try:
            response = await client.post(url, json=payload, headers=headers)
            elapsed_ms = (time.perf_counter() - started) * 1000
            try:
                body = response.json()
            except Exception:
                body = {"raw": response.text[:500]}
            usage = body.get("usage") if isinstance(body, dict) else {}
            return {
                "status_code": response.status_code,
                "latency_ms": elapsed_ms,
                "accepted": 200 <= response.status_code < 300,
                "rejected": response.status_code == 429,
                "prompt_tokens": int((usage or {}).get("prompt_tokens") or 0),
                "completion_tokens": int((usage or {}).get("completion_tokens") or 0),
                "session_id": metadata.get("x-session-id", "default"),
                "utility": float(metadata.get("x-request-utility", 1.0)),
                "policy": policy,
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return {
                "status_code": 0,
                "latency_ms": elapsed_ms,
                "accepted": False,
                "rejected": False,
                "error": str(exc),
                "policy": policy,
            }


async def run_experiment(
    target_url: str,
    workload: str,
    requests: int,
    concurrency: int,
    policy: str,
    output: str,
    seed: int,
    model: str | None,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    payloads = generate_requests(workload, requests, seed=seed, model=model)
    semaphore = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(300.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        tasks = [_send_one(client, target_url, payload, policy, semaphore) for payload in payloads]
        rows = await asyncio.gather(*tasks)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    latencies = [row["latency_ms"] for row in rows if row.get("accepted")]
    accepted = sum(1 for row in rows if row.get("accepted"))
    rejected = sum(1 for row in rows if row.get("rejected"))
    utility_goodput = sum(row.get("utility", 1.0) for row in rows if row.get("accepted"))
    summary = {
        "requests": requests,
        "accepted": accepted,
        "rejected": rejected,
        "accept_rate": accepted / max(1, requests),
        "utility_weighted_goodput": utility_goodput,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_mean_ms": statistics.mean(latencies) if latencies else 0.0,
        "runtime_s": time.perf_counter() - run_started,
        "output": str(output_path),
    }
    _update_manifest(
        output_path,
        summary,
        {
            "target_url": target_url,
            "workload": workload,
            "requests": requests,
            "concurrency": concurrency,
            "policy": policy,
            "seed": seed,
            "model": model,
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--workload", default="mixed_short_long")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--policy", default="infergate_admission")
    parser.add_argument("--output", default="results/smoke/client_results.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    asyncio.run(
        run_experiment(
            target_url=args.target_url,
            workload=args.workload,
            requests=args.requests,
            concurrency=args.concurrency,
            policy=args.policy,
            output=args.output,
            seed=args.seed,
            model=args.model,
        )
    )


if __name__ == "__main__":
    main()
