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

## Negative Result Policy

If InferGate does not improve utility-weighted goodput or session completion by at least 10% in an overloaded region, report the boundary condition explicitly and shift the narrative toward characterization and constrained-serving design tradeoffs.

