"""L'œil appris hors simulateur : la géométrie d'une détection, sans charger de réseau."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swarm_qr.detecteur import Detection  # noqa: E402


def test_une_detection_donne_son_centre_et_son_cote():
    d = Detection("qr", 0.9, np.array([100.0, 200.0, 160.0, 250.0]))
    assert np.allclose(d.centre, [130.0, 225.0])
    assert d.cote_px == 50.0
