"""Le contrôleur : amener un drone à une pose, l'y tenir, et le dire.

Il ne bloque jamais : à chaque appel de `tick`, il lit la position vraie, envoie une commande
de vitesse, et rend sa phase. C'est la boucle de mission qui fait avancer le monde, une fois
pour tous les drones — trois contrôleurs vivent donc dans la même boucle sans se gêner.

Phases : TRANSIT (vitesse constante de point de passage en point de passage), APPROCHE (vitesse
proportionnelle à la distance restante, plafonnée), TENUE (la pose est dans la tolérance et on
la garde un court instant), puis ATTEINT — ou ABANDON, avec sa raison : le délai est dépassé,
ou le drone ne progresse plus, ce qui est le cas d'un drone qui racle un rack.

Le contrôleur ne calcule pas de chemin : il reçoit les points de passage. Trouver un chemin
demande la carte des obstacles, qui n'existe qu'à l'étape 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

V_TRANSIT = 1.5       # m/s ; on ne lit pas en transit, seule l'inertie limite
V_APPROCHE = 1.0      # m/s ; la lecture ne souffre pas jusqu'à cette vitesse (étape 2)
GAIN = 0.9            # vitesse commandée = GAIN × distance restante
TOL = 0.15            # m ; tolérance d'arrivée
TOL_CAP = 0.09        # rad (5 degrés) ; la caméra doit regarder où il faut
GAIN_CAP = 1.5        # rad/s de rotation par rad d'écart de cap
VIT_CAP_MAX = 1.0     # rad/s
TOL_WP = 0.5          # m ; un point de passage se traverse, il ne se tient pas
TENUE_S = 0.5         # s ; deux à trois images à la cadence de mission
PATIENCE_S = 8.0      # s sans progrès avant d'abandonner
PROGRES_MIN = 0.10    # m ; en dessous, ce n'est pas un progrès
MARGE_BUDGET_S = 20.0

LOIS = ("proportionnel", "coupe", "autopilote")


class Phase(Enum):
    REPOS = "repos"
    TRANSIT = "transit"
    APPROCHE = "approche"
    TENUE = "tenue"
    ATTEINT = "atteint"
    ABANDON = "abandon"


TERMINALES = (Phase.ATTEINT, Phase.ABANDON)


@dataclass(frozen=True)
class Consigne:
    position: np.ndarray
    cap: float


@dataclass
class Bilan:
    phase: Phase
    raison: str
    t_transit: float
    t_approche: float
    t_tenue: float
    t_total: float
    longueur: float
    err_finale: float
    cap_err_finale: float
    v_finale: float
    n_points: int

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["phase"] = self.phase.value
        return d


def _ecart_cap_signe(a: float, b: float) -> float:
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def _ecart_cap(a: float, b: float) -> float:
    return abs(_ecart_cap_signe(a, b))


class Controleur:
    def __init__(self, pilot, get_pos, get_yaw=None, get_vel=None, *,
                 loi: str = "proportionnel", v_transit: float = V_TRANSIT,
                 v_approche: float = V_APPROCHE, gain: float = GAIN, tol: float = TOL,
                 tol_cap: float = TOL_CAP, tol_wp: float = TOL_WP, tenue_s: float = TENUE_S,
                 patience_s: float = PATIENCE_S):
        if loi not in LOIS:
            raise ValueError(f"loi inconnue : {loi}")
        self.pilot = pilot
        self.get_pos = get_pos
        self.get_yaw = get_yaw
        self.get_vel = get_vel
        self.loi = loi
        self.v_transit = v_transit
        self.v_approche = v_approche
        self.gain = gain
        self.tol = tol
        self.tol_cap = tol_cap
        self.tol_wp = tol_wp
        self.tenue_s = tenue_s
        self.patience_s = patience_s
        self.phase = Phase.REPOS
        self.consigne: Consigne | None = None
        self.bilan: Bilan | None = None
        self._tenue: Consigne | None = None          # au repos, aucune pose à tenir

    # --- affectation ---

    def assigne(self, consigne: Consigne, waypoints=(), budget_s: float | None = None) -> None:
        self.pilot.ensure_frame(self.get_pos, self.get_yaw)
        self.consigne = consigne
        self.points = [np.asarray(w, float) for w in waypoints]
        self.i_point = 0
        p = np.asarray(self.get_pos(), float)
        legs = [np.linalg.norm(q - a) for a, q in zip([p] + self.points, self.points)]
        dernier = float(np.linalg.norm(consigne.position - (self.points[-1] if self.points else p)))
        self.longueur = float(sum(legs) + dernier)
        if budget_s is None:
            budget_s = 2.0 * (sum(legs) / self.v_transit + dernier / self.v_approche) + MARGE_BUDGET_S
        self.budget_s = budget_s
        self.t0 = self.pilot.sim_clock
        self.t_fin_transit = None
        self.t_debut_tenue = None
        self.t_premiere_tenue = None
        self._coupe = False
        self._tenue = None
        vers = consigne.position - (self.points[-1] if self.points else p)
        self._axe = vers / max(np.linalg.norm(vers), 1e-6)
        self._reset_progres()
        self.bilan = None
        self.phase = Phase.TRANSIT if self.points else Phase.APPROCHE

    # --- un pas ---

    def tick(self) -> Phase:
        if self.phase in TERMINALES or self.consigne is None:
            if self._tenue is not None:
                self._tenir(self._tenue)
            return self.phase
        t = self.pilot.sim_clock
        p = np.asarray(self.get_pos(), float)
        if t - self.t0 > self.budget_s:
            return self._fin(Phase.ABANDON, "delai")

        if self.phase is Phase.TRANSIT:
            goal = self.points[self.i_point]
            err = goal - p
            d = float(np.linalg.norm(err))
            if d < self.tol_wp:
                self.i_point += 1
                self._reset_progres()
                if self.i_point == len(self.points):
                    self.phase = Phase.APPROCHE
                    self.t_fin_transit = t
                else:
                    goal = self.points[self.i_point]
                    err = goal - p
                    d = float(np.linalg.norm(err))
            if self.phase is Phase.TRANSIT:
                if not self._progresse(d, t):
                    return self._fin(Phase.ABANDON, "bloque")
                self._envoie(err / max(d, 1e-6) * self.v_transit, self.consigne.cap)
                return self.phase

        err = self.consigne.position - p
        d = float(np.linalg.norm(err))
        cap_ok = self.get_yaw is None or _ecart_cap(self.consigne.cap, self.get_yaw()) < self.tol_cap
        if self.phase is Phase.APPROCHE:
            if d > 2.0 * self.tol and not self._progresse(d, t):
                return self._fin(Phase.ABANDON, "bloque")
            if d < self.tol and cap_ok:
                self.phase = Phase.TENUE
                self.t_debut_tenue = t
                if self.t_premiere_tenue is None:
                    self.t_premiere_tenue = t
        elif self.phase is Phase.TENUE:
            if d > 1.5 * self.tol or not cap_ok:
                self.phase = Phase.APPROCHE
                self.t_debut_tenue = None
            elif t - self.t_debut_tenue >= self.tenue_s:
                return self._fin(Phase.ATTEINT, "")
        self._commande_approche(err, d)
        return self.phase

    # --- lois d'approche ---

    def _envoie(self, v: np.ndarray, cap: float) -> None:
        """Vitesse en monde, et le cap bouclé sur le cap VRAI : ArduPilot ne tient que le cap
        qu'il croit avoir, et son estimation s'écarte de la vérité de plusieurs degrés selon
        la direction (mesuré : 8,7 degrés). Une vitesse de rotation proportionnelle à l'écart
        vrai converge quel que soit ce biais. Sans cap vrai, on retombe sur le cap absolu."""
        if self.get_yaw is None:
            self.pilot.velocity_world(v, cap)
            return
        rotation = GAIN_CAP * _ecart_cap_signe(cap, self.get_yaw())
        self.pilot.velocity_world(v, yaw_rate_world=float(np.clip(rotation, -VIT_CAP_MAX, VIT_CAP_MAX)))

    def _commande_approche(self, err: np.ndarray, d: float) -> None:
        cap = self.consigne.cap
        if self.loi == "proportionnel":
            v = self.gain * err
            n = float(np.linalg.norm(v))
            if n > self.v_approche:
                v *= self.v_approche / n
            self._envoie(v, cap)
        elif self.loi == "coupe":
            # la commande la plus bête : pleine vitesse jusqu'à l'arrivée — ou jusqu'au
            # passage du plan de la cible, car un pas de 15 cm peut sauter la tolérance
            if self._coupe or d < self.tol or float(np.dot(err, self._axe)) <= 0.0:
                self._coupe = True
                self._envoie(np.zeros(3), cap)
            else:
                self._envoie(err / max(d, 1e-6) * self.v_approche, cap)
        else:
            self.pilot.position_world(self.consigne.position, cap)

    def _tenir(self, c: Consigne) -> None:
        """Garder une pose : une consigne de vitesse nulle laisse dériver le drone de 1 à 3 cm/s
        (mesuré), seule une loi de position le tient."""
        if self.loi == "autopilote":
            self.pilot.position_world(c.position, c.cap)
            return
        v = self.gain * (c.position - np.asarray(self.get_pos(), float))
        n = float(np.linalg.norm(v))
        if n > self.v_approche:
            v *= self.v_approche / n
        self._envoie(v, c.cap)

    # --- progrès et fin ---

    def _reset_progres(self) -> None:
        self._meilleure_d = math.inf
        self._t_meilleure = self.pilot.sim_clock

    def _progresse(self, d: float, t: float) -> bool:
        if d < self._meilleure_d - PROGRES_MIN:
            self._meilleure_d = d
            self._t_meilleure = t
        return t - self._t_meilleure <= self.patience_s

    def _fin(self, phase: Phase, raison: str) -> Phase:
        t = self.pilot.sim_clock
        p = np.asarray(self.get_pos(), float)
        t_transit = (self.t_fin_transit - self.t0) if self.t_fin_transit is not None else 0.0
        if self.t_premiere_tenue is not None:
            t_approche = self.t_premiere_tenue - self.t0 - t_transit
            t_tenue = t - self.t_premiere_tenue
        else:
            t_approche = t - self.t0 - t_transit
            t_tenue = 0.0
        self.bilan = Bilan(
            phase=phase, raison=raison,
            t_transit=round(t_transit, 2), t_approche=round(t_approche, 2),
            t_tenue=round(t_tenue, 2), t_total=round(t - self.t0, 2),
            longueur=round(self.longueur, 2),
            err_finale=round(float(np.linalg.norm(self.consigne.position - p)), 3),
            cap_err_finale=round(
                _ecart_cap(self.consigne.cap, self.get_yaw()) if self.get_yaw else 0.0, 3),
            v_finale=round(float(np.linalg.norm(self.get_vel())) if self.get_vel else 0.0, 3),
            n_points=len(self.points),
        )
        self.phase = phase
        # atteint : on tient la pose demandée ; abandon : on tient l'endroit où l'on s'est
        # arrêté, pour ne pas dériver vers un rack
        self._tenue = self.consigne if phase is Phase.ATTEINT else Consigne(p, self.consigne.cap)
        self._tenir(self._tenue)
        return phase


# ---------------------------------------------------------------- boucles

DT_TICK = 0.15


def pas(controleurs, clock, dt: float = DT_TICK) -> list[Phase]:
    """Un pas de mission : chaque drone envoie sa commande, puis le monde avance une fois."""
    phases = [c.tick() for c in controleurs]
    clock.pump(dt)
    return phases


def rejoindre(ctrl: Controleur, dt: float = DT_TICK, on_tick=None) -> Bilan:
    """Bloquant, pour un seul drone : avance jusqu'à ATTEINT ou ABANDON."""
    while True:
        phase = ctrl.tick()
        if on_tick is not None:
            on_tick(ctrl)
        if phase in TERMINALES:
            return ctrl.bilan
        ctrl.pilot.pump(dt)
