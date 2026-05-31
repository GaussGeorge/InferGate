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
.\scripts\smoke_test.ps1
```

The smoke test starts a mock OpenAI-compatible vLLM server and writes JSONL output to `results/smoke/`.

## Real vLLM

InferGate does not download models and does not require vLLM to run on Windows. Start vLLM in Docker, WSL, Linux, or a remote host, then point InferGate at it:

```powershell
$env:MODEL_ID="Qwen3.5-7B"
$env:VLLM_BASE_URL="http://127.0.0.1:8000"
$env:VLLM_METRICS_URL="http://127.0.0.1:8000/metrics"
$env:INFERGATE_POLICY="infergate_admission"
python -m uvicorn infergate.app:app --host 127.0.0.1 --port 8080
```

Run a real-service smoke level:

```powershell
.\scripts\smoke_test.ps1 -UseRealVllm -Requests 10
```

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

## Experiments

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

Trace records include:

```text
request_id, session_id, policy, decision, estimated_cost,
utility, session_step, queue_wait_ms, ttft_ms, e2e_ms,
prompt_tokens, completion_tokens, accepted, rejected, degraded
```

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

