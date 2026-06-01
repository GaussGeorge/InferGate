# InferGate Notes

## Working Title

InferGate: Utility- and Cache-Aware Admission Control for Resource-Constrained LLM Serving

## Contributions

1. Characterization: single-GPU vLLM deployments show effective completed value loss under FIFO/SJF/EDF/VTC-inspired baselines when load exceeds the A4000 16GB service envelope.
2. Design: utility-aware admission jointly models load, estimated token cost, SLO success probability, and session progress to avoid wasting scarce decode/prefill budget.
3. Cache-aware Warming: InferGate warms only high-value, high-reuse prefixes under an explicit token budget and only during low-load windows.
4. Evaluation: A4000 + vLLM + Qwen experiments report utility goodput, SLO satisfaction, TTFT/E2E latency, prefix hit behavior, and negative-result boundaries.

## System Boundary

InferGate is an OpenAI API-compatible sidecar in front of vLLM. It does not modify vLLM, LMCache, or VTC. VTC is used only as a VTC-inspired external fairness baseline based on accumulated tenant token cost.

## Evaluation Outline

Stage 1 validates the proxy and mock vLLM path. Stage 2 validates logging, policy switching, and stress stability. Stage 3 runs no-cache baselines. Stage 4 evaluates cache-aware warmup using vLLM APC or LMCache when available. Stage 5 packages figures, manifest, and reproduction instructions.

## Milestone 2 Real vLLM Smoke

Real smoke used vLLM on WSL, served to Windows as `http://127.0.0.1:9999`, with served model name `qwen` and model path `/mnt/d/model_path/qwen3.5-4b`. This is Qwen3.5-4B, not the earlier 7B target, and must be reported as such in manifests and paper text.

Launch command observed from WSL:

```bash
python -m vllm.entrypoints.openai.api_server --model /mnt/d/model_path/qwen3.5-4b --served-model-name qwen --trust-remote-code --host 127.0.0.1 --port 9999 --max-model-len 4096 --gpu-memory-utilization 0.8 --enforce-eager --max-num-seqs 8 --enable-auto-tool-choice --tool-call-parser hermes
```

Smoke outcome:

```text
10 requests, concurrency 1: 10/10 accepted, gateway P95 0.91 ms
100 requests, concurrency 4: 100/100 accepted, gateway P95 1.13 ms
300 requests, concurrency 8: 300/300 accepted, gateway P95 1.12 ms
```

The 1000-request smoke was reduced to 300 because the 100-request run took about 250 seconds on the A4000/Qwen3.5-4B setup; forcing 1000 requests would not add meaningful integration signal for this milestone.

Real metrics sample is saved under `results/real_vllm/metrics_sample.txt` and mirrored as a committed parser fixture in `tests/fixtures/real_vllm_metrics_sample.txt`. The sample includes `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, `vllm:prefix_cache_queries_total`, and `vllm:prefix_cache_hits_total`.

Tokenizer fallback rate was 100% during this smoke, so Milestone 3 should either install/use a HuggingFace tokenizer for `qwen` or explicitly keep the character-length estimator as an experimental approximation.

## Milestone 3 Real Metrics and Policy Calibration

Milestone 3 split the vLLM served model name from the tokenizer path:

```text
MODEL_ID=qwen
INFERGATE_TOKENIZER_ID=D:\model_path\qwen3.5-4b
INFERGATE_USE_HF_TOKENIZER=1
```

With `python -m pip install -e ".[dev,tokenizer]"`, the real 10-request smoke reached 10/10 accepted and tokenizer fallback rate 0%. The matching model paths are `/mnt/d/model_path/qwen3.5-4b` inside WSL and `D:\model_path\qwen3.5-4b` on Windows.

The frozen A4000 calibration settings for the next stage are:

```text
max_active_requests=4
max_queue_size=32
queue_timeout_ms=120000
kv_reject_threshold=0.80
waiting_reject_threshold=8
admission_reject_score=0.0030
admission_degrade_score=0.0060
degraded_max_tokens=64
```

Calibration ran `long_context`, `mixed_short_long`, and `agent_session` with concurrency 8/12/16 and 60 requests per run. All real calibration traces are under `results/calibration/`.

Observed overload boundary:

```text
long_context: accept + defer + degrade + reject at concurrency 8/12/16
mixed_short_long: accept + defer + degrade at concurrency 8/12/16
agent_session: accept + defer + degrade at concurrency 8/12/16
```

The strongest overload signal is `long_context`, where low-score high-cost requests are rejected with `low_score_overload`, mid-score requests are degraded with `degrade_low_score_overload`, and high-score requests are deferred with `high_score_queue`. Queue saturation, rather than vLLM waiting depth alone, is the first reliable pressure signal for this single-A4000 setup.

Gateway P95 across the calibration summary is below 5 ms after moving queue-state reads to a non-blocking snapshot. Degrade records satisfy `max_tokens_sent < max_tokens_original`.

## Milestone 4 Admission-Only Pilot

Pilot matrix completed on real A4000/vLLM/Qwen3.5-4B with:

```text
policies=fcfs,sjf,edf,static_threshold,vtc_inspired,infergate_admission
workloads=long_context,mixed_short_long,agent_session
concurrency=8,12,16
requests_per_run=20
repeats=1
```

The pilot produced 54 client traces, 54 InferGate traces, `results/main_pilot/summary.csv`, `results/main_pilot/decision_breakdown.csv`, and six figures under `paper/figures/main_pilot/`. Tokenizer fallback stayed at 0%, global gateway P95 stayed below 5 ms, and all degraded requests satisfied `max_tokens_sent < max_tokens_original`.

Pilot overload behavior:

```text
long_context: infergate_admission produces accept + defer + degrade + reject
mixed_short_long: infergate_admission produces accept + defer + degrade
agent_session: infergate_admission produces accept + defer + degrade
```

Intended claims for the main-matrix figures:

```text
utility_goodput_per_second.png: In high-contention long-context workloads, InferGate trades some rejection/degradation for higher utility-weighted completion rate per second.
slo_satisfaction_rate.png: Admission decisions expose the SLO boundary where FCFS-like policies keep accepting work that cannot complete within deadlines.
session_completion_rate.png: Session progress weighting protects multi-step agent sessions from mid-session loss under overload.
ttft_p95.png: Defer/degrade decisions reduce tail latency pressure for accepted work, especially on long-context and mixed workloads.
decision_breakdown.png: The calibrated policy produces explainable action diversity rather than accepting all requests.
degraded_and_rejected_rate.png: InferGate's cost is visible as controlled degradation and entry rejection, which must be compared against utility and session completion gains.
```

The 60-request, 3-repeat main matrix should reuse the frozen Milestone 3 thresholds without further tuning. `short_qa` remains a low-load boundary workload and can be omitted from the first required main pass if the three overload workloads already consume the available A4000 run window.

## Milestone 4 Admission-Only Main Matrix

Main matrix completed on real A4000/vLLM/Qwen3.5-4B using commit `c0c4e5a` and the frozen Milestone 3 calibrated thresholds. The first main pass intentionally focused on the three overload workloads and omitted `short_qa` as a low-load boundary workload:

```text
policies=fcfs,sjf,edf,static_threshold,vtc_inspired,infergate_admission
workloads=long_context,mixed_short_long,agent_session
concurrency=8,12,16
requests_per_run=60
repeats=3
total_runs=162
```

Artifacts:

```text
results/main_matrix/summary.csv
results/main_matrix/decision_breakdown.csv
results/main_matrix/matrix_manifest.jsonl
paper/figures/main_matrix/utility_goodput_per_second.png
paper/figures/main_matrix/slo_satisfaction_rate.png
paper/figures/main_matrix/session_completion_rate.png
paper/figures/main_matrix/ttft_p95.png
paper/figures/main_matrix/decision_breakdown.png
paper/figures/main_matrix/degraded_and_rejected_rate.png
```

Engineering validity checks:

```text
completed_runs=162/162
max_gateway_P95=3.04 ms
tokenizer_fallback_rate=0%
degrade_correct_rate=100%
infergate_decisions=accept 108, defer 1075, degrade 318, reject 119
```

Primary result:

```text
agent_session utility_goodput_per_second: +22.5% to +23.0% over the strongest baseline at concurrency 8/12, +5.0% at concurrency 16
long_context utility_goodput_per_second: +109% to +120% over the strongest baseline across concurrency 8/12/16
mixed_short_long utility_goodput_per_second: +19.5% to +19.9% over the strongest baseline across concurrency 8/12/16
```

Main claim for the paper: under resource-constrained A4000 serving, utility-aware admission improves utility goodput per second in overload workloads by rejecting or degrading low-score work and deferring high-score work, instead of preserving FIFO-style full acceptance.

Boundary and negative result:

```text
session_completion_rate does not improve over the best baseline in this matrix.
agent_session: InferGate is 1.0 at concurrency 8 and about 0.986 at concurrency 12/16, while several baselines remain at 1.0.
long_context: InferGate is about 0.78-0.79 because it rejects low-score entry requests; EDF-style baselines preserve 1.0 session completion by accepting everything, but with much lower utility goodput per second.
mixed_short_long: InferGate is about 0.994, slightly below the 1.0 best baseline.
```

Interpretation: the admission-only result should be framed as a utility-goodput improvement with an explicit completion-rate tradeoff, not as a universal session-completion win. The strongest claim region is `long_context` and `mixed_short_long`; `agent_session` is a weaker positive result for utility goodput and a boundary case for session completion.

Next step for Milestone 5: keep these admission thresholds frozen and test whether cache-aware warmup can reduce TTFT P95 or improve prefix-hit behavior without increasing warmup overhead above the 10% prompt-token budget.

## Negative Result Policy

If InferGate does not improve utility-weighted goodput or session completion by at least 10% in an overloaded region, report the boundary condition explicitly and shift the narrative toward characterization and constrained-serving design tradeoffs.
