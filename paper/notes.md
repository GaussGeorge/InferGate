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

## Negative Result Policy

If InferGate does not improve utility-weighted goodput or session completion by at least 10% in an overloaded region, report the boundary condition explicitly and shift the narrative toward characterization and constrained-serving design tradeoffs.
