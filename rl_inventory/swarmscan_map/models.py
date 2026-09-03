"""Acteur-critique rsl_rl : CNN sur les cartes égocentriques + MLP sur le vecteur,
critique privilégié (CTDE) — l'acteur ne voit JAMAIS le groupe « privileged »."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCritic


class _MapNet(nn.Module):
    """Découpe [maps | extra] en interne : CNN sur les cartes, MLP sur le reste."""

    def __init__(self, map_channels: int, map_px: int, extra_dim: int, out_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.map_dim = map_channels * map_px * map_px
        self.shape = (map_channels, map_px, map_px)
        self.cnn = nn.Sequential(
            nn.Conv2d(map_channels, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ELU(),
            nn.Flatten(),
            nn.Linear(64 * (map_px // 8) ** 2, 256), nn.ELU(),
        )
        layers, prev = [], 256 + extra_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maps = x[..., : self.map_dim].reshape(-1, *self.shape)
        extra = x[..., self.map_dim:]
        return self.head(torch.cat([self.cnn(maps), extra], dim=-1))


class MapActorCritic(ActorCritic):
    """Même interface publique qu'ActorCritic ; actor/critic remplacés par des _MapNet."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        map_channels: int = 18,
        map_px: int = 32,
        actor_hidden_dims=(256, 128),
        critic_hidden_dims=(256, 128),
        init_noise_std: float = 0.5,
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.obs_groups = obs_groups
        map_dim = map_channels * map_px * map_px

        actor_dim = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        critic_dim = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        self.actor = _MapNet(map_channels, map_px, actor_dim - map_dim, num_actions, tuple(actor_hidden_dims))
        self.critic = _MapNet(map_channels, map_px, critic_dim - map_dim, 1, tuple(critic_hidden_dims))
        nn.init.uniform_(self.actor.head[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.actor.head[-1].bias)

        self.actor_obs_normalization = False
        self.critic_obs_normalization = False
        self.actor_obs_normalizer = nn.Identity()
        self.critic_obs_normalizer = nn.Identity()

        self.noise_std_type = "scalar"
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        torch.distributions.Normal.set_default_validate_args(False)

    def _bounded_mean(self, obs) -> torch.Tensor:
        # moyenne bornée en douceur à ±3 : l'évasion hors de la zone de rappel (moyennes éjectées
        # par un update violent puis coincées dans le plat, 3 runs tués) devient impossible par
        # construction ; quasi-identité dans [−1,1] (écart max 3,5 %), gradient jamais nul
        return 3.0 * torch.tanh(self.actor(obs) / 3.0)

    def update_distribution(self, obs):
        mean = self._bounded_mean(obs)
        # σ PLAFONNÉ : sans borne, un accident d'update + bonus d'entropie → explosion (σ=9 vécu,
        # = la pathologie n°3 de l'audit IPPO ; remède max_log_std enfin appliqué à notre pile)
        std = self.std.clamp(0.05, 1.2).expand_as(mean)
        self.distribution = torch.distributions.Normal(mean, std)

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self._bounded_mean(obs)
