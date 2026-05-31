SYSTEM_PROMPTS = {
    "qa": "You are a concise technical assistant.",
    "rag": "You answer using the provided context. Cite only the supplied context.",
    "agent": "You are an agent executing a multi-step diagnostic workflow.",
}

LONG_CONTEXT = """
InferGate studies admission control for resource-constrained LLM serving. The
system runs as a sidecar in front of an OpenAI-compatible vLLM endpoint and
uses utility, SLO slack, session progress, estimated token cost, and prefix
cache reuse potential to decide whether requests should be accepted, deferred,
degraded, or rejected. The cache-aware path records prefix hashes and triggers
budgeted warmup only under low-load windows.
""".strip()

QUESTIONS = [
    "Summarize the main bottleneck in one sentence.",
    "Give two likely failure modes.",
    "Explain why admission control can improve goodput under overload.",
    "List three metrics that should be reported.",
    "What should happen if vLLM metrics are unavailable?",
]

