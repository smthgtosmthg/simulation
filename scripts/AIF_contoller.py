"""
Threaded multi-agent LiDAR belief-mapping + action planner CONTROLLER.

Integration:
    controller = ThreadedLiDARController(
        w=W, h=H,
        n_agents=N,
        initial_positions=((x0,y0), (x1,y1), ...),
        max_range=6,
    )

    # each env tick:
    actions = controller.step(observations)   # observations is tuple length N
    # send actions back to env

Agent IDs:
- Each agent has a stable integer id: agent.agent_id
- Order of observations/actions corresponds to agent_id by default (0..N-1)

IMPORTANT:
- This controller needs each agent’s (x,y) to update belief correctly.
- This version updates internal poses by assuming the environment executes the
  previously returned actions exactly.
- If the environment can block moves, we should call:
      controller.set_positions(true_positions_from_env)
  before controller.step(observations).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# Basic math utils
# ============================================================

L_MAX = 30.0

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def bernoulli_entropy(p: float) -> float:
    p = clamp(p, 1e-6, 1 - 1e-6)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))

def logit(p: float) -> float:
    p = clamp(p, 1e-6, 1 - 1e-6)
    return math.log(p / (1 - p))

def inv_logit(l: float) -> float:
    if l >= 0:
        z = math.exp(-l)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(l)
        return z / (1.0 + z)

def softmax_pick(candidates, T: float, rng: random.Random):
    """
    candidates: list of (name, dx, dy, G) where lower G is better.
    Samples action with P(a) ∝ exp(-G/T)
    """
    T = max(T, 1e-6)
    m = min(c[3] for c in candidates)  # stability shift
    weights = [math.exp(-(c[3] - m) / T) for c in candidates]
    s = sum(weights)
    r = rng.random() * s
    acc = 0.0
    for c, w in zip(candidates, weights):
        acc += w
        if acc >= r:
            return c
    return candidates[-1]

# ============================================================
# Directions and actions
# ============================================================

DIRS4 = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
DIR_LIST = ["N", "E", "S", "W"]

ACTIONS = [
    ("stay", 0, 0),
    ("N", 0, -1),
    ("NE", 1, -1),
    ("E", 1, 0),
    ("SE", 1, 1),
    ("S", 0, 1),
    ("SW", -1, 1),
    ("W", -1, 0),
    ("NW", -1, -1),
]

# ============================================================
# Belief map
# ============================================================

class BeliefMap:
    def __init__(self, w: int, h: int, p0: float = 0.5):
        self.w = w
        self.h = h
        self.p0 = p0
        self.l0 = logit(p0)
        self.p = [[p0 for _ in range(w)] for _ in range(h)]
        self.l = [[self.l0 for _ in range(w)] for _ in range(h)]

    def prob(self, x: int, y: int) -> float:
        return self.p[y][x]

    def update_logodds(self, x: int, y: int, delta_l: float):
        self.l[y][x] = clamp(self.l[y][x] + delta_l, -L_MAX, L_MAX)
        self.p[y][x] = inv_logit(self.l[y][x])

    def mean_entropy(self) -> float:
        tot = 0.0
        n = self.w * self.h
        for y in range(self.h):
            for x in range(self.w):
                tot += bernoulli_entropy(self.p[y][x])
        return tot / n

# ============================================================
# Belief fusion / mixing
# ============================================================

def fuse_beliefs(beliefs: List[BeliefMap], method: str = "logodds", weights=None) -> BeliefMap:
    assert len(beliefs) > 0
    w, h = beliefs[0].w, beliefs[0].h
    fused = BeliefMap(w, h, p0=beliefs[0].p0)
    l0 = fused.l0

    if weights is None:
        weights = [1.0] * len(beliefs)
    s = sum(weights)
    weights = [wi / s for wi in weights]

    if method == "avg":
        for y in range(h):
            for x in range(w):
                p = sum(b.p[y][x] * wi for b, wi in zip(beliefs, weights))
                fused.p[y][x] = p
                fused.l[y][x] = clamp(logit(p), -L_MAX, L_MAX)
        return fused

    if method != "logodds":
        raise ValueError(f"Unknown fusion method: {method}")

    for y in range(h):
        for x in range(w):
            l_sum = l0
            for b, wi in zip(beliefs, weights):
                l_sum += wi * (b.l[y][x] - l0)
            l_sum = clamp(l_sum, -L_MAX, L_MAX)
            fused.l[y][x] = l_sum
            fused.p[y][x] = inv_logit(l_sum)

    return fused

def mix_beliefs(local_b: BeliefMap, fused_b: BeliefMap, lam: float) -> BeliefMap:
    out = BeliefMap(local_b.w, local_b.h, p0=local_b.p0)
    for y in range(out.h):
        for x in range(out.w):
            l = (1 - lam) * local_b.l[y][x] + lam * fused_b.l[y][x]
            l = clamp(l, -L_MAX, L_MAX)
            out.l[y][x] = l
            out.p[y][x] = inv_logit(l)
    return out

# ============================================================
# Epistemic proxy
# ============================================================

def local_ray_expected_ig(bmap: BeliefMap, ax: int, ay: int, max_range: int) -> float:
    tot = 0.0
    for d in DIR_LIST:
        dx, dy = DIRS4[d]
        x, y = ax, ay
        p_reach = 1.0
        for _ in range(max_range):
            x += dx
            y += dy
            if x < 0 or y < 0 or x >= bmap.w or y >= bmap.h:
                break
            Hc = bernoulli_entropy(bmap.prob(x, y))
            tot += p_reach * Hc
            p_free = 1.0 - bmap.prob(x, y)
            p_reach *= p_free
            if p_reach < 1e-4:
                break
    return tot

# ============================================================
# Feasibility proxy (belief-based)
# ============================================================

def can_step_from_belief(
    belief: BeliefMap,
    x: int,
    y: int,
    dx: int,
    dy: int,
    occ_thr: float = 0.65,
) -> bool:
    nx, ny = x + dx, y + dy
    if nx < 0 or ny < 0 or nx >= belief.w or ny >= belief.h:
        return False
    if belief.prob(nx, ny) > occ_thr:
        return False
    if abs(dx) == 1 and abs(dy) == 1:
        if belief.prob(x + dx, y) > occ_thr or belief.prob(x, y + dy) > occ_thr:
            return False
    return True

# ============================================================
# Agent (controller-side)
# ============================================================

class Agent:
    def __init__(self, agent_id: int, x: int, y: int, max_range: int, seed: int):
        self.agent_id = int(agent_id)  # <-- ADDED stable id
        self.x = x
        self.y = y
        self.max_range = max_range
        self.rng = random.Random(seed)

    def lidar_predict_from_belief(self, belief: BeliefMap) -> List[float]:
        preds = []
        for d in DIR_LIST:
            dx, dy = DIRS4[d]
            x, y = self.x, self.y
            p_no_hit = 1.0
            for _ in range(self.max_range):
                x += dx
                y += dy
                if x < 0 or y < 0 or x >= belief.w or y >= belief.h:
                    p_no_hit *= 0.0
                    break
                p_no_hit *= (1.0 - belief.prob(x, y))
            preds.append(1.0 - p_no_hit)
        return preds

    def apply_action_assuming_env_executes(self, dx: int, dy: int, w: int, h: int):
        nx = clamp(self.x + dx, 0, w - 1)
        ny = clamp(self.y + dy, 0, h - 1)
        self.x, self.y = int(nx), int(ny)

# ============================================================
# Mapping update from external LiDAR distances
# ============================================================

def inverse_sensor_update_distance(
    belief: BeliefMap,
    agent: Agent,
    dists: Sequence[int],
    lo_free: float,
    lo_occ: float,
):
    ax, ay = agent.x, agent.y
    for i, dname in enumerate(DIR_LIST):
        dx, dy = DIRS4[dname]
        dist = int(dists[i])
        x, y = ax, ay

        free_steps = min(dist - 1, agent.max_range)
        for _ in range(free_steps):
            x += dx
            y += dy
            if 0 <= x < belief.w and 0 <= y < belief.h:
                belief.update_logodds(x, y, lo_free)

        if dist <= agent.max_range:
            x += dx
            y += dy
            if 0 <= x < belief.w and 0 <= y < belief.h:
                belief.update_logodds(x, y, lo_occ)

# ============================================================
# Planning / resilience params
# ============================================================

@dataclass
class Params:
    w_epistemic: float = 2.5
    w_entropy_recover: float = 3.0
    w_innov_recover: float = 1.2
    w_move: float = 0.0
    w_deadline: float = 12.0
    w_entropy_durable: float = 1.2
    w_innov_durable: float = 0.8
    w_churn_durable: float = 2.2
    w_maintain: float = 10.0
    H_target: float = 0.44
    innov_target: float = 0.16

@dataclass
class ResilienceState:
    stress_active: bool = False
    stress_t0: int = -1
    recovered_at: int = -1
    durable_count: int = 0

def plan_action_belief_only(
    belief: BeliefMap,
    agent: Agent,
    res: ResilienceState,
    t: int,
    alpha: int,
    beta: int,
    params: Params,
    horizon: int = 2,
    softmax_T: float = 0.25,
    rng: Optional[random.Random] = None,
    occ_thr: float = 0.65,
) -> Tuple[str, int, int, float]:
    if rng is None:
        rng = agent.rng

    step_rel0 = (t - res.stress_t0) if res.stress_active else 10**9
    candidates: List[Tuple[str, int, int, float]] = []

    for name, dx, dy in ACTIONS:
        if can_step_from_belief(belief, agent.x, agent.y, dx, dy, occ_thr=occ_thr):
            nx, ny = agent.x + dx, agent.y + dy
        else:
            nx, ny = agent.x, agent.y

        total_G = 0.0
        ax, ay = nx, ny

        for h in range(horizon):
            step_rel = step_rel0 + h

            ig_proxy = local_ray_expected_ig(belief, ax, ay, agent.max_range)
            epistemic_bonus = params.w_epistemic * ig_proxy

            H = belief.mean_entropy()

            tmp_agent = Agent(agent_id=-1, x=ax, y=ay, max_range=agent.max_range, seed=0)
            pred = tmp_agent.lidar_predict_from_belief(belief)
            innov_proxy = sum(p * (1 - p) for p in pred) / 4.0
            churn_proxy = innov_proxy

            move_cost = 0.0 if name == "stay" else 1.0

            if res.stress_active and step_rel <= alpha:
                G = (
                    params.w_entropy_recover * H
                    + params.w_innov_recover * innov_proxy
                    + params.w_move * move_cost
                    - epistemic_bonus
                )
                if step_rel == alpha and H > params.H_target:
                    G += params.w_deadline * (H - params.H_target)

            elif res.stress_active and (alpha < step_rel <= alpha + beta):
                maintain_pen = 0.0
                if H > params.H_target:
                    maintain_pen += (H - params.H_target)
                if innov_proxy > params.innov_target:
                    maintain_pen += (innov_proxy - params.innov_target)

                G = (
                    params.w_entropy_durable * H
                    + params.w_innov_durable * innov_proxy
                    + params.w_churn_durable * churn_proxy
                    + params.w_move * move_cost
                    + params.w_maintain * maintain_pen
                    - epistemic_bonus
                )
            else:
                G = (0.6 * H + 0.2 * innov_proxy + params.w_move * move_cost - epistemic_bonus)

            total_G += G

            if can_step_from_belief(belief, ax, ay, dx, dy, occ_thr=occ_thr):
                ax, ay = ax + dx, ay + dy

        total_G += rng.uniform(-1e-4, 1e-4)
        candidates.append((name, dx, dy, total_G))

    return softmax_pick(candidates, T=softmax_T, rng=rng)

# ============================================================
# Controller
# ============================================================

Obs4 = Tuple[int, int, int, int]
Pos2 = Tuple[int, int]
Act2 = Tuple[int, int]

class ThreadedLiDARController:
    """
    Input:  observations = tuple of per-agent observations (N,E,S,W distances)
    Output: actions      = tuple of per-agent actions (dx,dy), same order as agents
    """

    def __init__(
        self,
        w: int,
        h: int,
        n_agents: int,
        initial_positions: Tuple[Pos2, ...],
        max_range: int = 6,
        fusion: str = "logodds",
        plan_mix: float = 0.3,
        softmax_T: float = 0.25,
        horizon: int = 2,
        alpha: int = 200,
        beta: int = 600,
        seed: int = 1,
        workers: int = 0,
        lo_free: float = -0.55,
        lo_occ: float = 0.85,
        occ_thr_plan: float = 0.65,
        params: Optional[Params] = None,
    ):
        if len(initial_positions) != n_agents:
            raise ValueError("initial_positions must have length n_agents")

        self.w = w
        self.h = h
        self.n_agents = n_agents

        self.max_range = max_range
        self.fusion = fusion
        self.plan_mix = plan_mix
        self.softmax_T = softmax_T
        self.horizon = horizon
        self.alpha = alpha
        self.beta = beta

        self.lo_free = lo_free
        self.lo_occ = lo_occ
        self.occ_thr_plan = occ_thr_plan

        self.params = params if params is not None else Params()
        self.res = ResilienceState()

        self.t = 0

        self.agents: List[Agent] = []
        self.beliefs: List[BeliefMap] = []

        for i, (x, y) in enumerate(initial_positions):
            self.agents.append(Agent(agent_id=i, x=x, y=y, max_range=max_range, seed=seed * 1000 + i))
            self.beliefs.append(BeliefMap(w, h, p0=0.5))

        self.fused_belief = fuse_beliefs(
            self.beliefs,
            method=self.fusion,
            weights=[1.0 / self.n_agents] * self.n_agents
        )

        self.innov_ema = 0.0
        self.innov_var = 0.0
        self.ema_alpha = 0.05
        self.k_sigma = 2.0

        max_workers = workers if workers and workers > 0 else n_agents
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    @property
    def agent_ids(self) -> Tuple[int, ...]:
        """Convenience: stable ordering of agent ids (same order as observations/actions)."""
        return tuple(a.agent_id for a in self.agents)

    def close(self):
        self._pool.shutdown(wait=True)

    def set_positions(self, positions: Tuple[Pos2, ...]) -> None:
        if len(positions) != self.n_agents:
            raise ValueError("positions must have length n_agents")
        for i, (x, y) in enumerate(positions):
            self.agents[i].x = int(clamp(x, 0, self.w - 1))
            self.agents[i].y = int(clamp(y, 0, self.h - 1))

    def _update_one_agent(self, i: int, fused_snapshot: BeliefMap, obs_i: Sequence[int]) -> float:
        agent = self.agents[i]
        b = self.beliefs[i]

        pred_probs = agent.lidar_predict_from_belief(fused_snapshot)
        innov_i = sum(
            abs((int(d) <= agent.max_range) - p)
            for d, p in zip(obs_i, pred_probs)
        ) / 4.0

        inverse_sensor_update_distance(b, agent, obs_i, self.lo_free, self.lo_occ)
        return innov_i

    def _plan_one_agent(self, i: int, fused_snapshot: BeliefMap) -> Act2:
        agent = self.agents[i]
        plan_belief = mix_beliefs(self.beliefs[i], fused_snapshot, lam=self.plan_mix)

        _name, dx, dy, _G = plan_action_belief_only(
            belief=plan_belief,
            agent=agent,
            res=self.res,
            t=self.t,
            alpha=self.alpha,
            beta=self.beta,
            params=self.params,
            horizon=self.horizon,
            softmax_T=self.softmax_T,
            rng=agent.rng,
            occ_thr=self.occ_thr_plan,
        )
        return (dx, dy)

    def step(self, observations: Tuple[Obs4, ...]) -> Tuple[Act2, ...]:
        if len(observations) != self.n_agents:
            raise ValueError("observations must have length n_agents")
        for obs in observations:
            if len(obs) != 4:
                raise ValueError("each observation must be 4 distances for (N,E,S,W)")

        # A) parallel belief updates
        fused_snapshot = self.fused_belief
        futs = [
            self._pool.submit(self._update_one_agent, i, fused_snapshot, observations[i])
            for i in range(self.n_agents)
        ]
        innovations = [f.result() for f in futs]
        innov_mean = sum(innovations) / max(1, len(innovations))

        # innovation stats
        err = innov_mean - self.innov_ema
        self.innov_ema += self.ema_alpha * err
        self.innov_var += self.ema_alpha * ((err * err) - self.innov_var)
        sigma = math.sqrt(max(self.innov_var, 1e-8))
        spike = innov_mean > (self.innov_ema + self.k_sigma * sigma)

        # fuse
        self.fused_belief = fuse_beliefs(
            self.beliefs,
            method=self.fusion,
            weights=[1.0 / self.n_agents] * self.n_agents
        )
        Hmean = self.fused_belief.mean_entropy()

        # B) optional stress logic
        if not self.res.stress_active and spike:
            self.res.stress_active = True
            self.res.stress_t0 = self.t
            self.res.recovered_at = -1
            self.res.durable_count = 0

        if self.res.stress_active:
            recovered_now = (Hmean <= self.params.H_target) and (innov_mean <= 0.30)
            if self.res.recovered_at < 0:
                if recovered_now:
                    self.res.recovered_at = self.t
                    self.res.durable_count = 0
            else:
                if recovered_now:
                    self.res.durable_count += 1
                else:
                    self.res.durable_count = 0
                if self.res.durable_count >= self.beta:
                    self.res.stress_active = False

        # C) parallel planning
        fused_snapshot2 = self.fused_belief
        futs = [self._pool.submit(self._plan_one_agent, i, fused_snapshot2) for i in range(self.n_agents)]
        actions_list = [f.result() for f in futs]
        actions: Tuple[Act2, ...] = tuple(actions_list)

        # D) update internal poses (assumes env executes actions)
        for i, (dx, dy) in enumerate(actions):
            self.agents[i].apply_action_assuming_env_executes(dx, dy, self.w, self.h)

        self.t += 1
        return actions


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    controller = ThreadedLiDARController(
        w=10, h=10,
        n_agents=2,
        initial_positions=((2, 2), (7, 7)),
        max_range=6,
        seed=1,
    )

    print("agent ids:", controller.agent_ids)  # (0, 1)

    obs = ((3, 7, 2, 5), (6, 1, 6, 2))
    actions = controller.step(obs)
    print("actions:", actions)

    controller.close()