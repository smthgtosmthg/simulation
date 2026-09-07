"""Le contrôleur, testé sans simulateur : un faux pilote qui déplace un point à la vitesse
commandée, avec un petit retard, suffit pour vérifier les phases, l'abandon et les bilans."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swarm_qr import control  # noqa: E402
from swarm_qr.control import Consigne, Controleur, Phase  # noqa: E402


class FauxPilote:
    """Un point qui suit la vitesse commandée avec une constante de temps, et un cap qui tourne
    à vitesse bornée. `mur` : plan x = mur infranchissable, pour simuler un rack."""

    def __init__(self, mur: float | None = None, tau: float = 0.3):
        self.t = 0.0
        self.tau = tau
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.v_cmd = np.zeros(3)
        self.cap = 0.0
        self.cap_cmd = 0.0
        self.rate_cmd = None
        self.mur = mur
        self.positions_cmd = []

    @property
    def sim_clock(self):
        return self.t

    def ensure_frame(self, get_pos, get_yaw=None):
        pass

    def velocity_world(self, v, yaw=None, yaw_rate_world=None):
        self.v_cmd = np.asarray(v, float)
        if yaw_rate_world is not None:
            self.rate_cmd = float(yaw_rate_world)
        elif yaw is not None:
            self.cap_cmd = yaw
            self.rate_cmd = None

    def hold(self, yaw=None):
        self.velocity_world(np.zeros(3), yaw)

    def position_world(self, p, yaw):
        self.positions_cmd.append(np.asarray(p, float))
        self.v_cmd = np.clip(np.asarray(p, float) - self.p, -1.0, 1.0)
        self.cap_cmd = yaw

    def pump(self, dt):
        n = int(round(dt / 0.01))
        for _ in range(n):
            self.v += (self.v_cmd - self.v) * 0.01 / self.tau
            self.p = self.p + self.v * 0.01
            if self.mur is not None and self.p[0] > self.mur:
                self.p[0] = self.mur
                self.v[0] = 0.0
            if self.rate_cmd is not None:
                self.cap += np.clip(self.rate_cmd, -0.5, 0.5) * 0.01
            else:
                d = math.atan2(math.sin(self.cap_cmd - self.cap), math.cos(self.cap_cmd - self.cap))
                self.cap += np.clip(d, -0.5 * 0.01, 0.5 * 0.01)
            self.t += 0.01


def _ctrl(pilote, **kw):
    return Controleur(pilote, lambda: pilote.p.copy(), lambda: pilote.cap,
                      lambda: pilote.v.copy(), **kw)


def test_approche_directe_atteint():
    pil = FauxPilote()
    c = _ctrl(pil)
    c.assigne(Consigne(np.array([3.0, 1.0, 0.5]), 0.3))
    b = control.rejoindre(c)
    assert b.phase is Phase.ATTEINT
    assert b.err_finale < control.TOL
    assert b.cap_err_finale < control.TOL_CAP
    assert b.t_transit == 0.0 and b.n_points == 0
    assert b.t_tenue >= control.TENUE_S


def test_transit_par_points_puis_approche():
    pil = FauxPilote()
    c = _ctrl(pil)
    pts = [np.array([0.0, 4.0, 0.0]), np.array([5.0, 4.0, 0.0])]
    c.assigne(Consigne(np.array([5.0, 0.0, 0.0]), 0.0), pts)
    assert c.phase is Phase.TRANSIT
    b = control.rejoindre(c)
    assert b.phase is Phase.ATTEINT
    assert b.n_points == 2
    assert b.t_transit > 0.0
    assert abs(b.longueur - 13.0) < 1e-6


def test_abandon_bloque_contre_un_mur():
    pil = FauxPilote(mur=1.0)
    c = _ctrl(pil, patience_s=3.0)
    c.assigne(Consigne(np.array([4.0, 0.0, 0.0]), 0.0), budget_s=60.0)
    b = control.rejoindre(c)
    assert b.phase is Phase.ABANDON and b.raison == "bloque"
    assert b.t_total < 10.0


def test_abandon_delai():
    pil = FauxPilote(mur=1.0)
    c = _ctrl(pil, patience_s=100.0)
    c.assigne(Consigne(np.array([4.0, 0.0, 0.0]), 0.0), budget_s=5.0)
    b = control.rejoindre(c)
    assert b.phase is Phase.ABANDON and b.raison == "delai"


def test_cap_lent_ne_declenche_pas_l_abandon():
    pil = FauxPilote()
    c = _ctrl(pil, patience_s=2.0)
    c.assigne(Consigne(np.array([0.3, 0.0, 0.0]), 3.0))
    b = control.rejoindre(c)
    assert b.phase is Phase.ATTEINT
    assert b.cap_err_finale < control.TOL_CAP


def test_loi_coupe_depasse_et_ne_revient_pas():
    # inertie d'un vrai drone : depuis 1 m/s, le test 5 a mesuré 1,06 m de glissade
    pil = FauxPilote(tau=1.0)
    c = _ctrl(pil, loi="coupe")
    c.assigne(Consigne(np.array([4.0, 0.0, 0.0]), 0.0), budget_s=15.0)
    b = control.rejoindre(c)
    assert b.phase is Phase.ABANDON
    assert pil.p[0] > 4.0 + control.TOL
    assert np.allclose(pil.v_cmd, 0.0)


def test_loi_autopilote_envoie_une_position():
    pil = FauxPilote()
    c = _ctrl(pil, loi="autopilote")
    c.assigne(Consigne(np.array([2.0, 0.0, 0.0]), 0.0))
    b = control.rejoindre(c)
    assert b.phase is Phase.ATTEINT
    assert pil.positions_cmd and np.allclose(pil.positions_cmd[-1], [2.0, 0.0, 0.0])


def test_pas_fait_avancer_plusieurs_controleurs():
    class Horloge:
        def __init__(self, pilotes):
            self.pilotes = pilotes

        def pump(self, dt):
            for p in self.pilotes:
                p.pump(dt)

    pils = [FauxPilote(), FauxPilote()]
    ctrls = [_ctrl(pils[0]), _ctrl(pils[1])]
    ctrls[0].assigne(Consigne(np.array([2.0, 0.0, 0.0]), 0.0))
    ctrls[1].assigne(Consigne(np.array([-2.0, 1.0, 0.0]), 1.0))
    horloge = Horloge(pils)
    for _ in range(400):
        phases = control.pas(ctrls, horloge)
        if all(ph in control.TERMINALES for ph in phases):
            break
    assert all(c.phase is Phase.ATTEINT for c in ctrls)


def test_la_tenue_ramene_le_drone_apres_l_arrivee():
    pil = FauxPilote()
    c = _ctrl(pil)
    c.assigne(Consigne(np.array([2.0, 0.0, 0.0]), 0.0))
    control.rejoindre(c)
    assert c.phase is Phase.ATTEINT
    pil.p = pil.p + np.array([0.0, 0.5, 0.0])       # une rafale le pousse
    for _ in range(40):
        c.tick()
        pil.pump(control.DT_TICK)
    assert c.phase is Phase.ATTEINT
    assert np.linalg.norm(pil.p - [2.0, 0.0, 0.0]) < control.TOL


def test_l_abandon_tient_la_position_d_arret():
    pil = FauxPilote(mur=1.0)
    c = _ctrl(pil, patience_s=2.0)
    c.assigne(Consigne(np.array([4.0, 0.0, 0.0]), 0.0), budget_s=60.0)
    control.rejoindre(c)
    assert c.phase is Phase.ABANDON
    arret = pil.p.copy()
    for _ in range(40):
        c.tick()
        pil.pump(control.DT_TICK)
    assert np.linalg.norm(pil.p - arret) < control.TOL


def test_le_cap_converge_par_la_vitesse_de_rotation():
    pil = FauxPilote()
    c = _ctrl(pil)
    c.assigne(Consigne(np.array([1.0, 0.0, 0.0]), -2.5))
    b = control.rejoindre(c)
    assert b.phase is Phase.ATTEINT
    assert pil.rate_cmd is not None                      # le cap est commandé en rotation
    assert abs(math.atan2(math.sin(pil.cap + 2.5), math.cos(pil.cap + 2.5))) < control.TOL_CAP


def test_sans_cap_vrai_le_cap_absolu_est_envoye():
    pil = FauxPilote()
    c = Controleur(pil, lambda: pil.p.copy(), None, lambda: pil.v.copy())
    c.assigne(Consigne(np.array([1.0, 0.0, 0.0]), 0.7))
    b = control.rejoindre(c)
    assert b.phase is Phase.ATTEINT
    assert pil.rate_cmd is None and pil.cap_cmd == 0.7


def test_budget_par_defaut_couvre_le_trajet():
    pil = FauxPilote()
    c = _ctrl(pil)
    c.assigne(Consigne(np.array([10.0, 0.0, 0.0]), 0.0), [np.array([0.0, 6.0, 0.0])])
    attendu = 2.0 * (6.0 / control.V_TRANSIT + math.hypot(10.0, 6.0) / control.V_APPROCHE) + control.MARGE_BUDGET_S
    assert abs(c.budget_s - attendu) < 1e-6


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
