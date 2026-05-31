from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _barplot(df: pd.DataFrame, y: str, output: Path, title: str) -> None:
    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=df, x="policy", y=y)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()
    df = pd.read_csv(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        raise SystemExit("summary is empty")
    _barplot(df, "utility_weighted_goodput", output_dir / "utility_goodput.png", "Utility-weighted goodput")
    _barplot(df, "slo_satisfaction_rate", output_dir / "slo_satisfaction.png", "SLO satisfaction")
    _barplot(df, "e2e_p95_ms", output_dir / "e2e_p95.png", "E2E P95 latency")
    _barplot(df, "entry_rejection_rate", output_dir / "entry_rejection.png", "Entry rejection rate")
    print(f"wrote figures to {output_dir}")


if __name__ == "__main__":
    main()

