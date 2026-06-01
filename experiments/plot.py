from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _barplot(df: pd.DataFrame, y: str, output: Path, title: str, hue: str | None = "workload") -> None:
    plt.figure(figsize=(11, 5.2))
    if "policy" not in df.columns:
        raise SystemExit("summary must contain a policy column")
    kwargs = {"data": df, "x": "policy", "y": y}
    if hue and hue in df.columns:
        kwargs["hue"] = hue
    sns.barplot(**kwargs)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _cache_barplot(df: pd.DataFrame, y: str, output: Path, title: str, workloads: list[str] | None = None) -> None:
    data = df.copy()
    if workloads is not None and "workload" in data.columns:
        data = data[data["workload"].isin(workloads)]
    plt.figure(figsize=(12, 5.4))
    sns.barplot(data=data, x="cache_mode", y=y, hue="workload")
    plt.xticks(rotation=25, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _lineplot(df: pd.DataFrame, y: str, output: Path, title: str) -> None:
    plt.figure(figsize=(11, 5.2))
    sns.lineplot(data=df, x="concurrency", y=y, hue="policy", style="workload", markers=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _decision_breakdown(df: pd.DataFrame, output: Path) -> None:
    decision_cols = ["decision_accept", "decision_defer", "decision_degrade", "decision_reject"]
    available = [col for col in decision_cols if col in df.columns]
    if not available:
        return
    grouped = df.groupby(["policy"], as_index=False)[available].sum()
    melted = grouped.melt(id_vars=["policy"], value_vars=available, var_name="decision", value_name="count")
    melted["decision"] = melted["decision"].str.replace("decision_", "", regex=False)
    plt.figure(figsize=(11, 5.2))
    sns.barplot(data=melted, x="policy", y="count", hue="decision")
    plt.xticks(rotation=30, ha="right")
    plt.title("Decision breakdown")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _degraded_rejected(df: pd.DataFrame, output: Path) -> None:
    cols = [col for col in ["degraded_rate", "entry_rejection_rate", "non_2xx_rate"] if col in df.columns]
    if not cols:
        return
    grouped = df.groupby(["policy", "workload"], as_index=False)[cols].mean()
    melted = grouped.melt(id_vars=["policy", "workload"], value_vars=cols, var_name="rate_type", value_name="rate")
    plt.figure(figsize=(12, 5.8))
    sns.barplot(data=melted, x="policy", y="rate", hue="rate_type")
    plt.xticks(rotation=30, ha="right")
    plt.title("Degraded, rejected, and non-2xx rates")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def _cache_decision_breakdown(df: pd.DataFrame, output: Path) -> None:
    decision_cols = ["decision_accept", "decision_defer", "decision_degrade", "decision_reject"]
    available = [col for col in decision_cols if col in df.columns]
    if not available:
        return
    grouped = df.groupby(["cache_mode"], as_index=False)[available].sum()
    melted = grouped.melt(id_vars=["cache_mode"], value_vars=available, var_name="decision", value_name="count")
    melted["decision"] = melted["decision"].str.replace("decision_", "", regex=False)
    plt.figure(figsize=(11, 5.2))
    sns.barplot(data=melted, x="cache_mode", y="count", hue="decision")
    plt.xticks(rotation=25, ha="right")
    plt.title("Cache decision breakdown")
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
    if "prefix_cache_queries_delta" in df.columns and "cache_mode" in df.columns:
        _cache_barplot(df, "prefix_cache_hit_rate_delta", output_dir / "prefix_hit_rate_delta.png", "Prefix hit-rate delta")
        _cache_barplot(df, "TTFT_P95", output_dir / "ttft_p95_cache_modes.png", "TTFT P95 by cache mode")
        _cache_barplot(df, "warmup_token_overhead_ratio", output_dir / "warmup_overhead.png", "Warmup token overhead")
        _cache_barplot(df, "utility_goodput_per_second", output_dir / "utility_goodput_cache_modes.png", "Utility goodput by cache mode")
        _cache_barplot(
            df,
            "warmup_token_overhead_ratio",
            output_dir / "non_reuse_overhead.png",
            "Non-reuse warmup overhead",
            workloads=["non_reuse_control"],
        )
        _cache_decision_breakdown(df, output_dir / "cache_decision_breakdown.png")
    elif "utility_goodput_per_second" in df.columns:
        _barplot(df, "utility_goodput_per_second", output_dir / "utility_goodput_per_second.png", "Utility goodput per second")
        _barplot(df, "SLO_satisfaction_rate", output_dir / "slo_satisfaction_rate.png", "SLO satisfaction rate")
        _barplot(df, "session_completion_rate", output_dir / "session_completion_rate.png", "Session completion rate")
        _lineplot(df, "TTFT_P95", output_dir / "ttft_p95.png", "TTFT P95")
        _decision_breakdown(df, output_dir / "decision_breakdown.png")
        _degraded_rejected(df, output_dir / "degraded_and_rejected_rate.png")
    else:
        _barplot(df, "utility_weighted_goodput", output_dir / "utility_goodput.png", "Utility-weighted goodput", hue=None)
        _barplot(df, "slo_satisfaction_rate", output_dir / "slo_satisfaction.png", "SLO satisfaction", hue=None)
        _barplot(df, "e2e_p95_ms", output_dir / "e2e_p95.png", "E2E P95 latency", hue=None)
        _barplot(df, "entry_rejection_rate", output_dir / "entry_rejection.png", "Entry rejection rate", hue=None)
    print(f"wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
