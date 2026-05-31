import json

import httpx
import pytest

from experiments.mock_vllm_server import app as mock_app
from infergate.proxy import ProxyConfig, create_app
from infergate.schemas import LoadSnapshot, PolicySettings


@pytest.mark.asyncio
async def test_proxy_completes_request_against_mock_asgi(tmp_path) -> None:
    mock_transport = httpx.ASGITransport(app=mock_app)

    cfg = ProxyConfig(
        vllm_base_url="http://mock",
        metrics_url="http://mock/metrics",
        results_dir=str(tmp_path),
        policy_name="infergate_admission",
        settings=PolicySettings(max_active_requests=2),
    )
    app = create_app(cfg)

    async def local_forward(payload, headers=None, is_warmup=False):
        async with httpx.AsyncClient(transport=mock_transport, base_url="http://mock") as client:
            return await client.post("/v1/chat/completions", json=payload, headers=headers)

    async def local_load():
        return LoadSnapshot(num_requests_waiting=0, kv_cache_usage_perc=0.2, metrics_available=True)

    app.state.forwarder = local_forward
    app.state.load_snapshot_provider = local_load

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://infergate") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
                "metadata": {"x-request-utility": 2.0, "x-cache-key": "k"},
            },
            headers={"x-infergate-policy": "infergate_admission"},
        )
    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] == 8


@pytest.mark.asyncio
async def test_proxy_rejects_streaming_requests(tmp_path) -> None:
    cfg = ProxyConfig(
        vllm_base_url="http://mock",
        metrics_url="http://mock/metrics",
        results_dir=str(tmp_path),
        policy_name="infergate_admission",
    )
    app = create_app(cfg)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://infergate") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "unsupported_streaming"


@pytest.mark.asyncio
async def test_trace_has_top_level_score_and_reason(tmp_path) -> None:
    mock_transport = httpx.ASGITransport(app=mock_app)
    cfg = ProxyConfig(
        vllm_base_url="http://mock",
        metrics_url="http://mock/metrics",
        results_dir=str(tmp_path),
        policy_name="infergate_admission",
    )
    app = create_app(cfg)

    async def local_forward(payload, headers=None, is_warmup=False):
        async with httpx.AsyncClient(transport=mock_transport, base_url="http://mock") as client:
            return await client.post("/v1/chat/completions", json=payload, headers=headers)

    async def local_load():
        return LoadSnapshot(num_requests_waiting=0, kv_cache_usage_perc=0.2, metrics_available=True)

    app.state.forwarder = local_forward
    app.state.load_snapshot_provider = local_load
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://infergate") as client:
        await client.post(
            "/v1/chat/completions",
            json={"model": "mock", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 8},
        )
    trace_path = tmp_path / "infergate_trace.jsonl"
    record = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert "score" in record
    assert "reason" in record
    assert "gateway_ms" in record
    assert "cache_backend" in record
