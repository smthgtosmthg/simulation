from __future__ import annotations

import csv as _csv
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# Sentinelles pour identifier le "cloud" comme expéditeur/destinataire spécial
CLOUD_SRC: int = -99
CLOUD_DST_ALL: int = -1


class MessageQueue:

    def __init__(self):
        self._pending: List[Tuple[int, int, int, str, Any]] = []
        self.dropped: int = 0
        self.sent: int = 0
        self.delivered: int = 0
        self.sent_by_type: Dict[str, int] = {"belief": 0, "action": 0}
        self.dropped_by_type: Dict[str, int] = {"belief": 0, "action": 0}
        self.delivered_by_type: Dict[str, int] = {"belief": 0, "action": 0}

    def send(self, src: int, dst: int, msg_type: str, payload: Any,
             current_step: int, latency_ms: Optional[float],
             step_dt_ms: float) -> None:
        if latency_ms is None or latency_ms < 0:
            self.dropped += 1
            self.dropped_by_type[msg_type] = self.dropped_by_type.get(msg_type, 0) + 1
            return
        steps_delay = max(0, math.ceil(latency_ms / max(step_dt_ms, 1e-3)))
        arrival = current_step + steps_delay
        self._pending.append((arrival, src, dst, msg_type, payload))
        self.sent += 1
        self.sent_by_type[msg_type] = self.sent_by_type.get(msg_type, 0) + 1

    def deliver(self, current_step: int) -> List[Tuple[int, int, str, Any]]:
        ready: List[Tuple[int, int, str, Any]] = []
        remaining: List[Tuple[int, int, int, str, Any]] = []
        for entry in self._pending:
            arr, src, dst, typ, payload = entry
            if arr <= current_step:
                ready.append((src, dst, typ, payload))
            else:
                remaining.append(entry)
        self._pending = remaining
        self.delivered += len(ready)
        for _, _, typ, _ in ready:
            self.delivered_by_type[typ] = self.delivered_by_type.get(typ, 0) + 1
        return ready

    @property
    def size(self) -> int:
        return len(self._pending)

    def stats(self) -> Dict[str, Any]:
        return {
            "queue_size": self.size,
            "sent": self.sent,
            "delivered": self.delivered,
            "dropped": self.dropped,
            "sent_by_type": dict(self.sent_by_type),
            "delivered_by_type": dict(self.delivered_by_type),
            "dropped_by_type": dict(self.dropped_by_type),
        }


class NS3LatencyReader:

    WIFI_CSV = "/tmp/ns3_output.csv"
    LTE5G_CSV = "/tmp/drone_latency_ns3.csv"
    LTE5G_METRICS_CSV = "/tmp/drone_5g_metrics.csv"

    def __init__(self, mode: str = "none"):
        self.mode = mode
        self._warned_missing = False
        self.last_pairs: Dict[Tuple[int, int], Dict[str, float]] = {}

    @property
    def active(self) -> bool:
        return self.mode in ("wifi", "5g")

    def _csv_path(self) -> str:
        if self.mode == "wifi":
            return self.WIFI_CSV
        if self.mode == "5g":
            if os.path.exists(self.LTE5G_CSV):
                return self.LTE5G_CSV
            return self.LTE5G_METRICS_CSV
        return ""

    def read_latencies(self) -> Dict[Tuple[int, int], float]:
        if not self.active:
            return {}
        path = self._csv_path()
        if not path or not os.path.exists(path):
            if not self._warned_missing:
                print(f"  [NS-3] ⚠ Fichier latences absent ({path}) "
                      f"→ utilisation latence=0 (dégradation gracieuse)")
                self._warned_missing = True
            return {}
        latencies: Dict[Tuple[int, int], float] = {}
        pairs_full: Dict[Tuple[int, int], Dict[str, float]] = {}
        try:
            with open(path) as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    try:
                        i_key = "drone_i" if "drone_i" in row else "drone_a"
                        j_key = "drone_j" if "drone_j" in row else "drone_b"
                        i = int(row[i_key])
                        j = int(row[j_key])
                        lat = float(row.get("latency_ms", 0.0))
                        jitter = float(row.get("jitter_ms", 0.0)) if "jitter_ms" in row else 0.0
                        rx = int(row.get("rx_packets", 0)) if "rx_packets" in row else 0
                        key = (min(i, j), max(i, j))
                        latencies[key] = lat
                        pairs_full[key] = {
                            "latency_ms": round(lat, 3),
                            "jitter_ms": round(jitter, 3),
                            "rx_packets": rx,
                        }
                    except (ValueError, KeyError, TypeError):
                        continue
        except (IOError, OSError):
            return {}
        self.last_pairs = pairs_full
        return latencies

    def pair_latency(self, i: int, j: int, fallback: float = 0.0) -> float:
        key = (min(i, j), max(i, j))
        return self.last_pairs.get(key, {}).get("latency_ms", fallback)


@dataclass
class LinkState:

    cloud_link_active: bool = True
    cut_pairs: Set[frozenset] = field(default_factory=set)
    all_drone_links_cut: bool = False

    def is_link_cut(self, a: int, b: int) -> bool:
        if self.all_drone_links_cut:
            return True
        return frozenset({a, b}) in self.cut_pairs

    def cut_pairs_list(self) -> List[str]:
        return ["-".join(str(x) for x in sorted(p)) for p in self.cut_pairs]
