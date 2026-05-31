# InferGate

InferGate is an OpenAI API-compatible sidecar for vLLM serving experiments. It implements utility-aware and cache-aware admission control without modifying vLLM, LMCache, or VTC.

Default endpoints:

```text
InferGate: http://127.0.0.1:8080/v1/chat/completions
vLLM:      http://127.0.0.1:8000/v1/chat/completions
metrics:   http://127.0.0.1:8000/metrics
```

## Setup

```powershell
cd D:\InferGate
python -m pip install -e ".[dev]"
python -m pytest
```

`stream=true` is intentionally unsupported in this experiment build. InferGate returns HTTP 400 for streaming requests so traces keep complete TTFT/E2E/token accounting.

## Run Path 1: Mock Smoke

This path requires no model and no vLLM installation. It starts a mock OpenAI-compatible vLLM server plus InferGate, sends 20 requests, then writes JSONL output to `results/smoke/`.

```powershell
.\scripts\smoke_test.ps1
```

Expected outputs:

```text
results/smoke/client_results.jsonl
results/smoke/infergate_trace.jsonl
results/manifest.json
```

Trace records include top-level `policy`, `decision`, `score`, `reason`, `estimated_cost`, `gateway_ms`, `ttft_ms`, `e2e_ms`, `cache_backend`, and `tokenizer_fallback`.

## Run Path 2: Real vLLM Smoke

InferGate does not download models and does not require vLLM to run on Windows. Start vLLM in Docker, WSL, Linux, or a remote host, then point InferGate at it:

```powershell
$env:MODEL_ID="Qwen3.5-7B"
$env:VLLM_BASE_URL="http://127.0.0.1:8000"
$env:VLLM_METRICS_URL="http://127.0.0.1:8000/metrics"
$env:INFERGATE_POLICY="infergate_admission"
python -m uvicorn infergate.app:app --host 127.0.0.1 --port 8080
```

Recommended vLLM flags for the A4000 target:

```text
--enable-prefix-caching
--gpu-memory-utilization 0.85
--max-model-len 8192
```

If A4000 memory is insufficient, lower `--max-model-len` to `4096`.

Run real-service smoke levels:

```powershell
.\scripts\smoke_test.ps1 -UseRealVllm -Requests 10
.\scripts\smoke_test.ps1 -UseRealVllm -Requests 100
.\scripts\smoke_test.ps1 -UseRealVllm -Requests 1000
```

The 10-request run should be 100% successful. For 100/1000-request runs, inspect `results/smoke/infergate_trace.jsonl` for `vllm_unreachable`, `queue_timeout`, timeout, or OOM symptoms.

## Policies

Each policy implements:

```python
decide(request, load_snapshot, queue_state, cache_state) -> Decision
```

Available policies:

```text
fcfs
sjf
edf
static_threshold
vtc_inspired
infergate_admission
infergate_cache
```

Per-request metadata can be passed through OpenAI `metadata` or headers:

```text
x-session-id
x-session-step
x-session-total-steps
x-request-utility
x-request-deadline-ms
x-cache-key
```

Missing fields default to `utility=1`, `step=0`, `deadline=None`, and `cache_key=hash(prompt prefix)`.

## Cache Behavior

`infergate_cache` records prefix hash observations and can issue warmup requests when:

```text
num_requests_waiting == 0
kv_cache_usage_perc < 0.65
warmup_token_budget_used < 0.10 * total_prompt_tokens
predicted_reuse_count >= 2
```

Warmups use `metadata={"infergate_warmup": true}`, `max_tokens=1`, and are tracked separately. If LMCache is configured but metrics are unavailable, InferGate marks results as `cache_backend=vllm_apc`.

## Run Path 3: Main Matrix

Single run:

```powershell
python -m experiments.run_experiment --policy infergate_admission --workload mixed_short_long --requests 1000 --concurrency 32 --output results/main/run.jsonl
```

Stage 3 matrix:

```powershell
python -m experiments.run_all --requests 1000 --output-dir results/main
python -m experiments.summarize --input-dir results/main --output results/summary.csv
python -m experiments.plot --summary results/summary.csv --output-dir paper/figures
```

`experiments.run_all` is the admission-only Stage 3 entry point. It writes raw JSONL files under `results/main/`; `summarize` produces CSV aggregates; `plot` regenerates figures from raw results.

## Trace Schema

Trace records include:

```text
request_id, session_id, policy, decision, estimated_cost,
utility, session_step, queue_wait_ms, ttft_ms, e2e_ms,
prompt_tokens, completion_tokens, accepted, rejected, degraded,
score, reason, gateway_ms, cache_backend, tokenizer_fallback
```

## Artifact Reproduction

1. Install the project with `python -m pip install -e ".[dev]"`.
2. Run `python -m pytest`.
3. Run `.\scripts\smoke_test.ps1` for a no-model mock smoke.
4. For real vLLM, set `MODEL_ID`, `VLLM_BASE_URL`, and `VLLM_METRICS_URL`, then run the three real smoke levels.
5. Run `python -m experiments.summarize --input-dir results --output results/summary.csv`.
6. Run `python -m experiments.plot --summary results/summary.csv --output-dir paper/figures`.

`results/manifest.json` records seed, policy, workload, request count, concurrency, model, output path, runtime summary, and commit hash when Git metadata is available.

## Artifact Layout

```text
configs/        A4000, policy, workload configuration
infergate/      sidecar implementation
baselines/      baseline policy shims
workloads/      synthetic workload generator
experiments/    runners, summary, plots, mock vLLM
scripts/        vLLM examples and smoke test
paper/          notes, literature matrix, figures
tests/          unit and mock integration tests
```
