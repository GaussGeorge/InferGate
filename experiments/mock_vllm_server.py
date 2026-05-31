from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from infergate.tokenizer import TokenEstimator, extract_prompt_text

app = FastAPI(title="Mock vLLM OpenAI Server")
estimator = TokenEstimator()
state = {
    "waiting": 0,
    "running": 0,
    "cache_usage": 0.25,
}


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True}


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    body = "\n".join(
        [
            "# HELP vllm:num_requests_waiting Number of waiting requests.",
            "# TYPE vllm:num_requests_waiting gauge",
            f"vllm:num_requests_waiting {state['waiting']}",
            "# HELP vllm:num_requests_running Number of running requests.",
            "# TYPE vllm:num_requests_running gauge",
            f"vllm:num_requests_running {state['running']}",
            "# HELP vllm:gpu_cache_usage_perc GPU KV cache usage.",
            "# TYPE vllm:gpu_cache_usage_perc gauge",
            f"vllm:gpu_cache_usage_perc {state['cache_usage']}",
        ]
    )
    return PlainTextResponse(body + "\n")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict[str, Any]:
    payload = await request.json()
    prompt = extract_prompt_text(payload)
    prompt_tokens = estimator.count(prompt)
    max_tokens = int(payload.get("max_tokens", payload.get("max_completion_tokens", 16)) or 16)
    completion_tokens = min(max_tokens, 32)
    state["running"] += 1
    state["cache_usage"] = min(0.90, state["cache_usage"] + prompt_tokens / 200_000)
    try:
        await asyncio.sleep(min(0.20, 0.005 + prompt_tokens / 50_000))
        return {
            "id": f"chatcmpl-mock-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model") or "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "mock completion",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    finally:
        state["running"] = max(0, state["running"] - 1)
        state["cache_usage"] = max(0.10, state["cache_usage"] * 0.999)

