"""PS-PPO : aplatit le DirectMARLEnv (3 drones × B envs) en un VecEnv rsl_rl de 3B pseudo-envs.

Un seul cerveau partagé voit 3× plus de données ; les groupes d'observation
(maps / vector / privileged) sont découpés ici pour le runner rsl_rl 3.0.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from ..env import AGENTS
from .config_map import MAP_CFG
from .env_map import PRIV_DIM, VEC_DIM, SwarmScanMapEnv

MAP_DIM = MAP_CFG.map.obs_dim


class SwarmMapVecEnv(VecEnv):
    """Pseudo-env i = drone (i // B) dans l'entrepôt (i % B)."""

    def __init__(self, env: SwarmScanMapEnv):
        self.env = env
        self.B = env.num_envs
        self.num_envs = self.B * len(AGENTS)
        self.num_actions = env.cfg.action_spaces[AGENTS[0]]
        self.device = env.device
        self.cfg = {}
        self._obs = None

    @property
    def max_episode_length(self) -> int:
        return int(self.env.max_episode_length)  # dynamique : le budget d'épisode croît avec le niveau

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf.repeat(len(AGENTS))

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.env.episode_length_buf = value[: self.B]

    def _pack(self, obs_dict: dict) -> TensorDict:
        flat = torch.cat([obs_dict[a] for a in AGENTS], dim=0)
        return TensorDict(
            {
                "maps": flat[:, :MAP_DIM],
                "vector": flat[:, MAP_DIM : MAP_DIM + VEC_DIM],
                "privileged": flat[:, MAP_DIM + VEC_DIM : MAP_DIM + VEC_DIM + PRIV_DIM],
            },
            batch_size=[self.num_envs],
        )

    def get_observations(self) -> TensorDict:
        if self._obs is None:
            obs_dict, _ = self.env.reset()
            self._obs = self._pack(obs_dict)
        return self._obs

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        act = {a: actions[i * self.B : (i + 1) * self.B] for i, a in enumerate(AGENTS)}
        obs_dict, rew, terminated, truncated, extras = self.env.step(act)
        self._obs = self._pack(obs_dict)
        rewards = torch.cat([rew[a] for a in AGENTS], dim=0)
        time_outs = torch.cat([truncated[a] for a in AGENTS], dim=0)
        dones = torch.cat([(terminated[a] | truncated[a]) for a in AGENTS], dim=0)
        infos = {"time_outs": time_outs}
        if self.env.map_log:  # émis UNE fois quand frais (répété, il biaisait les moyennes TensorBoard)
            infos["log"] = dict(self.env.map_log)
            self.env.map_log = {}
        return self._obs, rewards, dones.float(), infos

    def reset(self):
        obs_dict, _ = self.env.reset()
        self._obs = self._pack(obs_dict)
        return self._obs

    def close(self):
        self.env.close()
