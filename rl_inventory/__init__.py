"""rl_inventory — Contrôleur RL pour l'inspection d'inventaire QR par drone.

Arène Isaac Lab + cerveaux Pore / PPO / Dreamer.
Voir CONCEPTION_controleur_RL_inventaire.md et README.md.

Enregistre la tâche essaim multi-drone. L'entry_point est résolu paresseusement
par gym.make (env.py n'est importé qu'après le lancement de l'app Isaac), donc
importer ce package reste léger (utile pour config_rl en python pur).
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-QR-Inventory-Swarm-Direct-v0",
    entry_point="rl_inventory.env:SwarmQREnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rl_inventory.env:SwarmQREnvCfg",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
