from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .run_experiment import run_experiment


async def main_async(args: argparse.Namespace) -> None:
    policies = args.policies.split(",")
    workloads = args.workloads.split(",")
    concurrencies = [int(item) for item in args.concurrency.split(",")]
    for policy in policies:
        for workload in workloads:
            for concurrency in concurrencies:
                for repeat in range(args.repeats):
                    output = (
                        Path(args.output_dir)
                        / f"policy={policy}"
                        / f"workload={workload}"
                        / f"concurrency={concurrency}"
                        / f"repeat={repeat}.jsonl"
                    )
                    await run_experiment(
                        target_url=args.target_url,
                        workload=workload,
                        requests=args.requests,
                        concurrency=concurrency,
                        policy=policy,
                        output=str(output),
                        seed=args.seed + repeat,
                        model=args.model,
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--policies", default="fcfs,sjf,edf,static_threshold,vtc_inspired,infergate_admission")
    parser.add_argument("--workloads", default="short_qa,long_context,mixed_short_long,agent_session")
    parser.add_argument("--concurrency", default="8,16,32,64")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", default="results/main")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

