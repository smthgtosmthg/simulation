"""
Réseau : file de messages avec latence, lecteur de latences NS-3, état des liens.

3 classes complémentaires :

- **MessageQueue** : FIFO horodatée. Un message envoyé au step t avec
  latency_ms arrive au step t + ceil(latency_ms / step_dt_ms). Si la latence
  est None ou < 0, le message est droppé (compté pour les métriques).
  Supporte deux types de payload :
    * "belief" — la croyance brute d'un drone (envoyée à un voisin ou au cloud)
    * "action" — une commande d'action calculée par le cloud, à envoyer à un drone

- **NS3LatencyReader** : lit le CSV produit par le scénario NS-3 et expose les
  latences inter-drone (paires i, j).  Cache pour le dashboard.

- **LinkState** : drapeaux booléens du réseau — cloud_link_active, cut_pairs,
  all_drone_links_cut.  Modifié par le StressorScheduler.
"""

from __future__ import annotations

import csv as _csv
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ════════════════════════════════════════════════════════════════════
# MessageQueue — Latence et drop des messages
# ════════════════════════════════════════════════════════════════════


# Sentinelles pour identifier le "cloud" comme expéditeur/destinataire spécial
CLOUD_SRC: int = -99
CLOUD_DST_ALL: int = -1


class MessageQueue:
    """File FIFO de messages réseau avec latence et drop.

    Un message a un type ("belief" ou "action") et un payload.  L'arrivée est
    calculée en steps AIF : arrival_step = current_step + ceil(latency / step_dt).
    """

    def __init__(self):
        # entries: list of (arrival_step, src, dst, type, payload)
        self._pending: List[Tuple[int, int, int, str, Any]] = []
        self.dropped: int = 0
        self.sent: int = 0
        self.delivered: int = 0
        # compteurs par type (pour debug / dashboard)
        self.sent_by_type: Dict[str, int] = {"belief": 0, "action": 0}
        self.dropped_by_type: Dict[str, int] = {"belief": 0, "action": 0}
        self.delivered_by_type: Dict[str, int] = {"belief": 0, "action": 0}

    def send(self, src: int, dst: int, msg_type: str, payload: Any,
             current_step: int, latency_ms: Optional[float],
             step_dt_ms: float) -> None:
        """Programme un envoi. Si latency_ms est None ou <0 → drop."""
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
        """Retourne et retire les messages dont arrival <= current_step."""
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


# ════════════════════════════════════════════════════════════════════
# NS3LatencyReader — Lecture des latences inter-drones depuis NS-3
# ════════════════════════════════════════════════════════════════════


class NS3LatencyReader:
    """Lit le CSV produit par NS-3 et expose les latences par paire (i, j)."""

    WIFI_CSV = "/tmp/ns3_output.csv"
    LTE5G_CSV = "/tmp/drone_latency_ns3.csv"
    LTE5G_METRICS_CSV = "/tmp/drone_5g_metrics.csv"

    def __init__(self, mode: str = "none"):
        self.mode = mode  # "none" | "wifi" | "5g"
        self._warned_missing = False
        # cache exposé au dashboard / aux artefacts
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


# ════════════════════════════════════════════════════════════════════
# LinkState — État booléen du réseau (modifié par les Stressors)
# ════════════════════════════════════════════════════════════════════


@dataclass
class LinkState:
    """Drapeaux de connectivité réseau.

    Cloud_link_active : True si le lien drone↔cloud est ouvert.
    cut_pairs         : ensemble de paires drone↔drone coupées (frozenset).
    all_drone_links_cut : True si tous les liens drone↔drone sont coupés
                          (raccourci pour cut_drone_link="all").
    """

    cloud_link_active: bool = True
    cut_pairs: Set[frozenset] = field(default_factory=set)
    all_drone_links_cut: bool = False

    def is_link_cut(self, a: int, b: int) -> bool:
        if self.all_drone_links_cut:
            return True
        return frozenset({a, b}) in self.cut_pairs

    def cut_pairs_list(self) -> List[str]:
        return ["-".join(str(x) for x in sorted(p)) for p in self.cut_pairs]
