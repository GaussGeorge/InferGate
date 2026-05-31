from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import Any

from .datasets import get_workload
from .session_templates import LONG_CONTEXT, QUESTIONS, SYSTEM_PROMPTS


def _utility(rng: random.Random, skew: float) -> float:
    if skew <= 1.0:
        return round(rng.uniform(0.5, 2.0), 3)
    return round(0.5 + min(8.0, rng.paretovariate(skew)), 3)


def generate_requests(
    workload: str,
    count: int,
    seed: int = 7,
    model: str | None = None,
) -> list[dict[str, Any]]:
    spec = get_workload(workload)
    rng = random.Random(seed)
    shared_prefix = f"{SYSTEM_PROMPTS['rag']}\n\nContext:\n{LONG_CONTEXT}"
    requests: list[dict[str, Any]] = []
    session_ids = [f"sess-{idx}" for idx in range(max(1, count // 5))]
    for idx in range(count):
        use_long = rng.random() < spec.long_context_ratio
        use_shared = rng.random() < spec.shared_prefix_ratio
        use_session = rng.random() < spec.session_ratio
        session_id = rng.choice(session_ids) if use_session else f"single-{idx}"
        session_step = rng.randint(0, 4) if use_session else 0
        total_steps = 5 if use_session else 1
        system_prompt = shared_prefix if use_shared else SYSTEM_PROMPTS["qa"]
        if use_long and not use_shared:
            system_prompt = f"{SYSTEM_PROMPTS['rag']}\n\nContext:\n{LONG_CONTEXT}\nUnique note {uuid.uuid4().hex}"
        question = rng.choice(QUESTIONS)
        utility = _utility(rng, spec.utility_skew)
        deadline_ms = rng.choice([None, 1500, 3000, 6000, 12000])
        cache_key = f"{workload}:shared" if use_shared else f"{workload}:{idx}"
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": 0,
            "max_tokens": rng.choice([32, 64, 128, 256]),
            "metadata": {
                "x-session-id": session_id,
                "x-session-step": session_step,
                "x-session-total-steps": total_steps,
                "x-request-utility": utility,
                "x-cache-key": cache_key,
            },
        }
        if deadline_ms is not None:
            body["metadata"]["x-request-deadline-ms"] = deadline_ms
        requests.append(body)
    return requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="mixed_short_long")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = generate_requests(args.workload, args.count, args.seed, args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

