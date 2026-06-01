from __future__ import annotations

import asyncio
import copy
import os
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .cache_registry import CacheRegistry
from .metrics import JsonlWriter, conservative_load_snapshot, fetch_load_snapshot
from .policies import make_policy
from .queue import AdmissionQueue
from .schemas import Action, Decision, LoadSnapshot, PolicySettings, TraceRecord
from .tokenizer import TokenEstimator, build_request_context, get_max_tokens
from .warmup import WarmupManager


class ProxyConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    vllm_base_url: str = "http://127.0.0.1:8000"
    metrics_url: str = "http://127.0.0.1:8000/metrics"
    model_id: str | None = None
    tokenizer_id: str | None = None
    policy_name: str = "infergate_admission"
    cache_backend: str = "vllm_apc"
    cache_mode: str = "no_cache_control"
    results_dir: str = "results/smoke"
    trace_filename: str = "infergate_trace.jsonl"
    prefer_hf_tokenizer: bool = False
    request_timeout_s: float = 300.0
    metrics_ttl_ms: int = 100
    settings: PolicySettings = PolicySettings()

    @classmethod
    def from_env(cls) -> "ProxyConfig":
        settings = PolicySettings(
            max_queue_size=int(os.getenv("INFERGATE_MAX_QUEUE_SIZE", "128")),
            max_active_requests=int(os.getenv("INFERGATE_MAX_ACTIVE", "8")),
            queue_timeout_ms=int(os.getenv("INFERGATE_QUEUE_TIMEOUT_MS", "120000")),
            kv_reject_threshold=float(os.getenv("INFERGATE_KV_REJECT_THRESHOLD", "0.92")),
            waiting_reject_threshold=int(os.getenv("INFERGATE_WAITING_REJECT_THRESHOLD", "64")),
            admission_reject_score=float(os.getenv("INFERGATE_REJECT_SCORE", "0.0010")),
            admission_degrade_score=float(os.getenv("INFERGATE_DEGRADE_SCORE", "0.0020")),
            degraded_max_tokens=int(os.getenv("INFERGATE_DEGRADED_MAX_TOKENS", "128")),
            vtc_tenant_token_limit=int(os.getenv("INFERGATE_VTC_TENANT_TOKEN_LIMIT", "60000")),
            warmup_budget_fraction=float(os.getenv("INFERGATE_WARMUP_BUDGET_FRACTION", "0.10")),
        )
        return cls(
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000"),
            metrics_url=os.getenv("VLLM_METRICS_URL", "http://127.0.0.1:8000/metrics"),
            model_id=os.getenv("MODEL_ID"),
            tokenizer_id=os.getenv("INFERGATE_TOKENIZER_ID"),
            policy_name=os.getenv("INFERGATE_POLICY", "infergate_admission"),
            cache_backend=os.getenv("CACHE_BACKEND", "vllm_apc"),
            cache_mode=os.getenv("INFERGATE_CACHE_MODE", "no_cache_control"),
            results_dir=os.getenv("INFERGATE_RESULTS_DIR", "results/smoke"),
            prefer_hf_tokenizer=os.getenv("INFERGATE_USE_HF_TOKENIZER", "0") == "1",
            request_timeout_s=float(os.getenv("INFERGATE_REQUEST_TIMEOUT_S", "300")),
            metrics_ttl_ms=int(os.getenv("INFERGATE_METRICS_TTL_MS", "100")),
            settings=settings,
        )


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _filtered_headers(request: Request) -> dict[str, str]:
    skip = {"host", "content-length", "connection"}
    return {key: value for key, value in request.headers.items() if key.lower() not in skip}


def _apply_degrade(payload: dict[str, Any], decision: Decision) -> dict[str, Any]:
    body = copy.deepcopy(payload)
    current = body.get("max_tokens", body.get("max_completion_tokens", 512))
    try:
        current_tokens = max(1, int(current))
    except (TypeError, ValueError):
        current_tokens = 512
    target = decision.degrade_max_tokens or max(1, current_tokens // 2)
    if target >= current_tokens:
        target = max(1, current_tokens // 2)
    body["max_tokens"] = max(1, min(current_tokens, target))
    metadata = body.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["infergate_degraded"] = True
        metadata["infergate_degrade_reason"] = decision.reason
    return body


def _extract_usage(response_json: dict[str, Any], fallback_prompt_tokens: int) -> tuple[int, int]:
    usage = response_json.get("usage") if isinstance(response_json.get("usage"), dict) else {}
    prompt_tokens = int(usage.get("prompt_tokens") or fallback_prompt_tokens)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return prompt_tokens, completion_tokens


def create_app(config: ProxyConfig | None = None) -> FastAPI:
    cfg = config or ProxyConfig.from_env()

    async def metrics_refresh_loop(lifespan_app: FastAPI) -> None:
        while True:
            snapshot = await fetch_load_snapshot(
                lifespan_app.state.config.metrics_url,
                client=lifespan_app.state.metrics_client,
            )
            lifespan_app.state.load_snapshot = snapshot
            lifespan_app.state.load_snapshot_ts = time.perf_counter()
            await asyncio.sleep(max(0.05, lifespan_app.state.config.metrics_ttl_ms / 1000.0))

    @asynccontextmanager
    async def lifespan(lifespan_app: FastAPI):
        lifespan_app.state.metrics_task = asyncio.create_task(metrics_refresh_loop(lifespan_app))
        try:
            yield
        finally:
            lifespan_app.state.metrics_task.cancel()
            with suppress(asyncio.CancelledError):
                await lifespan_app.state.metrics_task
            await lifespan_app.state.forward_client.aclose()
            await lifespan_app.state.metrics_client.aclose()

    app = FastAPI(title="InferGate", version="0.1.0", lifespan=lifespan)
    app.state.config = cfg
    app.state.token_estimator = TokenEstimator(cfg.tokenizer_id, prefer_hf=cfg.prefer_hf_tokenizer)
    app.state.cache_registry = CacheRegistry(cache_backend=cfg.cache_backend)
    app.state.warmup_manager = WarmupManager(
        cfg.model_id,
        cfg.settings.warmup_budget_fraction,
        cooldown_s=float(os.getenv("INFERGATE_WARMUP_COOLDOWN_S", "60")),
    )
    app.state.queue = AdmissionQueue(max_active_requests=cfg.settings.max_active_requests)
    app.state.trace_writer = JsonlWriter(Path(cfg.results_dir) / cfg.trace_filename)
    app.state.tenant_token_debt = {}
    app.state.forward_url = chat_completions_url(cfg.vllm_base_url)
    app.state.forward_client = httpx.AsyncClient(timeout=cfg.request_timeout_s, trust_env=False)
    app.state.metrics_client = httpx.AsyncClient(timeout=0.5, trust_env=False)
    app.state.load_snapshot = conservative_load_snapshot()
    app.state.load_snapshot_ts = 0.0

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "policy": app.state.config.policy_name,
            "vllm_url": app.state.forward_url,
            "cache_backend": app.state.cache_registry.cache_backend,
            "cache_mode": app.state.config.cache_mode,
            "model_id": app.state.config.model_id,
            "tokenizer_id": app.state.config.tokenizer_id,
            "tokenizer_fallback": app.state.token_estimator.fallback,
        }

    @app.get("/infergate/metrics")
    async def infergate_metrics() -> dict[str, Any]:
        queue_state = app.state.queue.snapshot_nowait(app.state.tenant_token_debt)
        warmup = app.state.warmup_manager.stats
        return {
            "queue": queue_state.model_dump(),
            "cache_backend": app.state.cache_registry.cache_backend,
            "cache_mode": app.state.config.cache_mode,
            "total_prompt_tokens": app.state.cache_registry.total_prompt_tokens,
            "warmup_token_budget_used": app.state.cache_registry.warmup_token_budget_used,
            "warmup_requests": warmup.warmup_requests,
            "warmup_tokens": warmup.warmup_tokens,
        }

    async def forward_to_vllm(payload: dict[str, Any], headers: dict[str, str] | None = None, is_warmup: bool = False) -> httpx.Response:
        custom_forwarder = getattr(app.state, "forwarder", None)
        if custom_forwarder is not None:
            return await custom_forwarder(payload, headers=headers, is_warmup=is_warmup)
        if cfg.model_id and not payload.get("model"):
            payload = copy.deepcopy(payload)
            payload["model"] = cfg.model_id
        if is_warmup:
            payload = copy.deepcopy(payload)
            payload.setdefault("metadata", {})["infergate_warmup"] = True
        return await app.state.forward_client.post(app.state.forward_url, json=payload, headers=headers)

    async def get_load_snapshot() -> LoadSnapshot:
        custom_load_provider = getattr(app.state, "load_snapshot_provider", None)
        if custom_load_provider is not None:
            return await custom_load_provider()
        return app.state.load_snapshot

    def trace_context(
        load_snapshot: LoadSnapshot,
        queue_state: Any,
        max_tokens_original: int,
        max_tokens_sent: int | None,
    ) -> dict[str, Any]:
        return {
            "load_running": load_snapshot.num_requests_running,
            "load_waiting": load_snapshot.num_requests_waiting,
            "load_kv_cache_usage": load_snapshot.kv_cache_usage_perc,
            "load_prefix_cache_hit_rate": load_snapshot.prefix_cache_hit_rate,
            "queue_active": queue_state.active_requests,
            "queue_waiting": queue_state.waiting_requests,
            "queue_saturation": queue_state.saturation,
            "max_tokens_original": max_tokens_original,
            "max_tokens_sent": max_tokens_sent,
            "prefix_cache_hit_rate_before": load_snapshot.prefix_cache_hit_rate,
            "prefix_cache_hit_rate_after": app.state.load_snapshot.prefix_cache_hit_rate,
        }

    async def maybe_schedule_warmup(load_snapshot: LoadSnapshot) -> tuple[bool, bool]:
        cache_mode = app.state.config.cache_mode
        if cache_mode == "no_cache_control":
            return False, False
        if app.state.cache_registry.cache_backend not in {"vllm_apc", "lmcache"}:
            return False, False
        if not load_snapshot.metrics_available:
            return False, False
        candidates = app.state.cache_registry.candidates(min_reuse_count=2)
        if cache_mode == "lru_warm":
            candidates = sorted(candidates, key=lambda item: item.last_seen_ts or 0.0, reverse=True)
        elif cache_mode == "always_warm":
            candidates = sorted(candidates, key=lambda item: item.last_warmup_ts or 0.0)
        elif cache_mode != "infergate_cache":
            return False, False
        saw_candidate = bool(candidates)
        for entry in candidates[:1]:
            if not app.state.warmup_manager.should_warmup(
                entry,
                load_snapshot,
                app.state.cache_registry.warmup_token_budget_used,
                app.state.cache_registry.total_prompt_tokens,
            ):
                continue
            app.state.cache_registry.mark_warmup(entry.cache_key, warmup_tokens=1)
            asyncio.create_task(app.state.warmup_manager.run_warmup(entry, forward_to_vllm))
            return True, True
        return saw_candidate, False

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        queue_wait_ms = 0.0
        payload = await request.json()
        request_started = time.perf_counter()
        headers = _filtered_headers(request)
        ctx = build_request_context(payload, request.headers, app.state.token_estimator)
        policy_name = request.headers.get("x-infergate-policy", app.state.config.policy_name)
        load_snapshot = await get_load_snapshot()
        if not load_snapshot.metrics_available and app.state.config.cache_backend == "lmcache":
            app.state.cache_registry.cache_backend = "vllm_apc"
        queue_state = await app.state.queue.snapshot(app.state.tenant_token_debt)
        if payload.get("stream") is True:
            gateway_ms = (time.perf_counter() - request_started) * 1000
            record = TraceRecord(
                request_id=ctx.request_id,
                session_id=ctx.session_id,
                policy=policy_name,
                decision=Action.REJECT.value,
                score=0.0,
                reason="streaming_not_supported",
                estimated_cost=ctx.estimated_cost,
                utility=ctx.utility,
                session_step=ctx.session_step,
                deadline_ms=ctx.deadline_ms,
                session_total_steps=ctx.session_total_steps,
                queue_wait_ms=0.0,
                gateway_ms=gateway_ms,
                **trace_context(load_snapshot, queue_state, ctx.max_tokens, None),
                prompt_tokens=ctx.prompt_tokens,
                accepted=False,
                rejected=True,
                degraded=False,
                step0_rejection=False,
                cache_key=ctx.cache_key,
                cache_backend=app.state.cache_registry.cache_backend,
                cache_mode=app.state.config.cache_mode,
                tokenizer_fallback=ctx.tokenizer_fallback,
                error="stream=true is not supported by InferGate experiments",
            )
            app.state.trace_writer.write(record.model_dump())
            return JSONResponse(
                {
                    "error": {
                        "message": "InferGate does not support stream=true in this experiment build",
                        "type": "unsupported_streaming",
                    }
                },
                status_code=400,
            )
        try:
            policy = make_policy(policy_name, app.state.config.settings)
        except KeyError:
            return JSONResponse({"error": {"message": f"unknown policy: {policy_name}"}}, status_code=400)

        cache_state = app.state.cache_registry.state(ctx.cache_key)
        decision = policy.decide(ctx, load_snapshot, queue_state, cache_state)
        gateway_ms = (time.perf_counter() - request_started) * 1000

        if decision.action == Action.REJECT:
            record = TraceRecord(
                request_id=ctx.request_id,
                session_id=ctx.session_id,
                policy=policy_name,
                decision=decision.action.value,
                score=decision.score,
                reason=decision.reason,
                estimated_cost=ctx.estimated_cost,
                utility=ctx.utility,
                session_step=ctx.session_step,
                deadline_ms=ctx.deadline_ms,
                session_total_steps=ctx.session_total_steps,
                queue_wait_ms=0.0,
                gateway_ms=gateway_ms,
                **trace_context(load_snapshot, queue_state, ctx.max_tokens, None),
                prompt_tokens=ctx.prompt_tokens,
                accepted=False,
                rejected=True,
                degraded=False,
                step0_rejection=ctx.session_step == 0,
                cache_key=ctx.cache_key,
                cache_backend=app.state.cache_registry.cache_backend,
                cache_mode=app.state.config.cache_mode,
                tokenizer_fallback=ctx.tokenizer_fallback,
                extra={"reason": decision.reason, "score": decision.score},
            )
            app.state.trace_writer.write(record.model_dump())
            return JSONResponse(
                {
                    "error": {
                        "message": "InferGate rejected request",
                        "type": "infergate_admission",
                        "reason": decision.reason,
                        "score": decision.score,
                    }
                },
                status_code=429,
            )

        body = _apply_degrade(payload, decision) if decision.action == Action.DEGRADE else copy.deepcopy(payload)
        max_tokens_sent = get_max_tokens(body)
        acquired = False
        e2e_started = time.perf_counter()
        response_json: dict[str, Any] | None = None
        status_code = 502
        error: str | None = None
        try:
            queue_wait_ms = await app.state.queue.acquire_or_queue(
                ctx.request_id,
                decision.priority,
                app.state.config.settings.queue_timeout_ms,
                force_queue=decision.action == Action.DEFER,
            )
            acquired = True
            upstream_started = time.perf_counter()
            upstream_response = await forward_to_vllm(body, headers=headers)
            ttft_ms = (time.perf_counter() - upstream_started) * 1000
            status_code = upstream_response.status_code
            content_type = upstream_response.headers.get("content-type", "application/json")
            try:
                response_json = upstream_response.json()
            except Exception:
                response_json = None
            e2e_ms = (time.perf_counter() - e2e_started) * 1000
            prompt_tokens, completion_tokens = (
                _extract_usage(response_json, ctx.prompt_tokens) if response_json is not None else (ctx.prompt_tokens, 0)
            )
            cache_hit = upstream_response.headers.get("x-cache-hit", "").lower() == "true"
            app.state.cache_registry.observe(
                ctx.cache_key,
                prompt_tokens=prompt_tokens,
                utility=ctx.utility,
                prompt_text=ctx.prompt_text,
                cache_hit=cache_hit,
            )
            app.state.tenant_token_debt[ctx.tenant_id] = (
                app.state.tenant_token_debt.get(ctx.tenant_id, 0) + prompt_tokens + completion_tokens
            )
            warmup_candidate, warmup_sent = await maybe_schedule_warmup(load_snapshot)
            record = TraceRecord(
                request_id=ctx.request_id,
                session_id=ctx.session_id,
                policy=policy_name,
                decision=decision.action.value,
                score=decision.score,
                reason=decision.reason,
                estimated_cost=ctx.estimated_cost,
                utility=ctx.utility,
                session_step=ctx.session_step,
                deadline_ms=ctx.deadline_ms,
                session_total_steps=ctx.session_total_steps,
                queue_wait_ms=queue_wait_ms,
                ttft_ms=ttft_ms,
                e2e_ms=e2e_ms,
                gateway_ms=gateway_ms,
                **trace_context(load_snapshot, queue_state, ctx.max_tokens, max_tokens_sent),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                accepted=200 <= status_code < 300,
                rejected=False,
                degraded=decision.action == Action.DEGRADE,
                cache_key=ctx.cache_key,
                cache_backend=app.state.cache_registry.cache_backend,
                cache_mode=app.state.config.cache_mode,
                warmup_candidate=decision.warmup_candidate or warmup_candidate,
                warmup_sent=warmup_sent,
                warmup_tokens_used=app.state.cache_registry.warmup_token_budget_used,
                tokenizer_fallback=ctx.tokenizer_fallback,
                error=None if 200 <= status_code < 500 else upstream_response.text[:500],
                extra={"reason": decision.reason, "score": decision.score},
            )
            app.state.trace_writer.write(record.model_dump())
            if response_json is not None and "application/json" in content_type:
                return JSONResponse(response_json, status_code=status_code)
            return Response(
                content=upstream_response.content,
                status_code=status_code,
                media_type=content_type.split(";")[0],
            )
        except asyncio.TimeoutError:
            error = "queue_timeout"
            status_code = 429
            return JSONResponse({"error": {"message": "InferGate queue timeout", "type": error}}, status_code=429)
        except httpx.RequestError as exc:
            error = f"vllm_unreachable: {exc}"
            status_code = 502
            return JSONResponse({"error": {"message": error, "type": "upstream_unreachable"}}, status_code=502)
        finally:
            if acquired:
                await app.state.queue.release()
            if error:
                record = TraceRecord(
                    request_id=ctx.request_id,
                    session_id=ctx.session_id,
                    policy=policy_name,
                    decision=decision.action.value,
                    score=decision.score,
                    reason=decision.reason,
                    estimated_cost=ctx.estimated_cost,
                    utility=ctx.utility,
                    session_step=ctx.session_step,
                    deadline_ms=ctx.deadline_ms,
                    session_total_steps=ctx.session_total_steps,
                    queue_wait_ms=queue_wait_ms,
                    e2e_ms=(time.perf_counter() - e2e_started) * 1000,
                    gateway_ms=gateway_ms,
                    **trace_context(load_snapshot, queue_state, ctx.max_tokens, max_tokens_sent),
                    prompt_tokens=ctx.prompt_tokens,
                    accepted=False,
                    rejected=status_code == 429,
                    degraded=decision.action == Action.DEGRADE,
                    step0_rejection=status_code == 429 and ctx.session_step == 0,
                    cache_key=ctx.cache_key,
                    cache_backend=app.state.cache_registry.cache_backend,
                    cache_mode=app.state.config.cache_mode,
                    tokenizer_fallback=ctx.tokenizer_fallback,
                    error=error,
                    extra={"reason": decision.reason, "score": decision.score},
                )
                app.state.trace_writer.write(record.model_dump())

    return app
