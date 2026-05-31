from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .schemas import LoadSnapshot

METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


@dataclass(frozen=True)
class MetricSample:
    name: str
    labels: dict[str, str]
    value: float


def parse_prometheus_metrics(text: str) -> list[MetricSample]:
    samples: list[MetricSample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        labels = {
            key: value.replace(r"\"", '"').replace(r"\\", "\\")
            for key, value in LABEL_RE.findall(match.group("labels") or "")
        }
        samples.append(
            MetricSample(
                name=match.group("name"),
                labels=labels,
                value=float(match.group("value")),
            )
        )
    return samples


def _sum_matching(samples: list[MetricSample], fragments: tuple[str, ...]) -> float:
    return sum(sample.value for sample in samples if all(fragment in sample.name for fragment in fragments))


def _max_matching(samples: list[MetricSample], fragments: tuple[str, ...]) -> float:
    values = [sample.value for sample in samples if all(fragment in sample.name for fragment in fragments)]
    return max(values) if values else 0.0


def load_snapshot_from_prometheus(text: str) -> LoadSnapshot:
    samples = parse_prometheus_metrics(text)
    waiting = _sum_matching(samples, ("request", "waiting")) or _sum_matching(samples, ("num_requests_waiting",))
    running = _sum_matching(samples, ("request", "running")) or _sum_matching(samples, ("num_requests_running",))
    kv_usage = (
        _max_matching(samples, ("kv_cache_usage",))
        or _max_matching(samples, ("gpu_cache_usage_perc",))
        or _max_matching(samples, ("cache_usage_perc",))
    )
    if kv_usage > 1.0:
        kv_usage = kv_usage / 100.0
    return LoadSnapshot(
        num_requests_waiting=int(waiting),
        num_requests_running=int(running),
        kv_cache_usage_perc=max(0.0, min(1.0, kv_usage)),
        gpu_cache_usage_perc=max(0.0, min(1.0, kv_usage)),
        metrics_available=True,
        warmup_allowed=True,
    )


def conservative_load_snapshot() -> LoadSnapshot:
    return LoadSnapshot(
        num_requests_waiting=0,
        num_requests_running=0,
        kv_cache_usage_perc=0.80,
        gpu_cache_usage_perc=0.80,
        metrics_available=False,
        warmup_allowed=False,
    )


async def fetch_load_snapshot(
    metrics_url: str,
    timeout_s: float = 0.5,
    client: httpx.AsyncClient | None = None,
) -> LoadSnapshot:
    try:
        if client is not None:
            response = await client.get(metrics_url, timeout=timeout_s)
        else:
            async with httpx.AsyncClient(timeout=timeout_s, trust_env=False) as local_client:
                response = await local_client.get(metrics_url)
        response.raise_for_status()
        return load_snapshot_from_prometheus(response.text)
    except Exception:
        return conservative_load_snapshot()


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, payload: dict[str, Any]) -> None:
        record = dict(payload)
        record.setdefault("ts", time.time())
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]
