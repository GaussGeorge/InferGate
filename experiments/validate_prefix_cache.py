from __future__ import annotations

import argparse
import asyncio
import csv
import os
from pathlib import Path
from typing import Any

import httpx

from infergate.metrics import fetch_load_snapshot
from workloads.generator import generate_requests


async def _send_requests(url: str, payloads: list[dict[str, Any]], concurrency: int) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(300.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        async def send_one(payload: dict[str, Any]) -> None:
            async with semaphore:
                response = await client.post(url, json=payload)
                response.raise_for_status()

        await asyncio.gather(*(send_one(payload) for payload in payloads))


async def _fetch_metrics_text(metrics_url: str) -> str:
    async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
        response = await client.get(metrics_url)
        response.raise_for_status()
        return response.text


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _prefix_cache_enabled(metrics_text: str) -> bool | None:
    for line in metrics_text.splitlines():
        if line.startswith("vllm:cache_config_info") and "enable_prefix_caching=" in line:
            if 'enable_prefix_caching="True"' in line:
                return True
            if 'enable_prefix_caching="False"' in line:
                return False
    return None


def _reason(queries_delta: float, hits_delta: float, prefix_cache_enabled: bool | None) -> str:
    if prefix_cache_enabled is False:
        return "vllm_prefix_caching_disabled_restart_with_enable_prefix_caching"
    if queries_delta <= 0:
        return "prefix_cache_queries_not_increasing"
    if hits_delta <= 0:
        return "queries_increased_but_hits_not_increasing_check_apc_prompt_template_or_model_config"
    return "prefix_cache_counters_increase"


async def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.vllm_base_url or os.getenv("VLLM_BASE_URL", "http://127.0.0.1:9999")
    metrics_url = args.metrics_url or os.getenv("VLLM_METRICS_URL", f"{base_url.rstrip('/')}/metrics")
    model = args.model or os.getenv("MODEL_ID", "qwen")
    payloads = generate_requests(args.workload, args.requests, seed=args.seed, model=model)
    before_text = await _fetch_metrics_text(metrics_url)
    (output_dir / "metrics_before.txt").write_text(before_text, encoding="utf-8")
    before = await fetch_load_snapshot(metrics_url)
    await _send_requests(_chat_url(base_url), payloads, args.concurrency)
    after_text = await _fetch_metrics_text(metrics_url)
    (output_dir / "metrics_after.txt").write_text(after_text, encoding="utf-8")
    after = await fetch_load_snapshot(metrics_url)
    queries_before = before.prefix_cache_queries_total or 0.0
    queries_after = after.prefix_cache_queries_total or 0.0
    hits_before = before.prefix_cache_hits_total or 0.0
    hits_after = after.prefix_cache_hits_total or 0.0
    row = {
        "workload": args.workload,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "model": model,
        "vllm_base_url": base_url,
        "metrics_url": metrics_url,
        "kv_cache_usage_before": before.kv_cache_usage_perc,
        "kv_cache_usage_after": after.kv_cache_usage_perc,
        "prefix_cache_queries_before": queries_before,
        "prefix_cache_queries_after": queries_after,
        "prefix_cache_queries_delta": queries_after - queries_before,
        "prefix_cache_hits_before": hits_before,
        "prefix_cache_hits_after": hits_after,
        "prefix_cache_hits_delta": hits_after - hits_before,
        "prefix_cache_hit_rate_before": before.prefix_cache_hit_rate,
        "prefix_cache_hit_rate_after": after.prefix_cache_hit_rate,
        "prefix_cache_enabled": _prefix_cache_enabled(after_text),
        "reason": _reason(queries_after - queries_before, hits_after - hits_before, _prefix_cache_enabled(after_text)),
    }
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"wrote {summary_path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/cache_validation")
    parser.add_argument("--workload", default="repeated_rag_context")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default=None)
    parser.add_argument("--vllm-base-url", default=None)
    parser.add_argument("--metrics-url", default=None)
    args = parser.parse_args()
    asyncio.run(run_validation(args))


if __name__ == "__main__":
    main()
