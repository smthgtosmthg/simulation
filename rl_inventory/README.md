# rl_inventory — Contrôleur RL d'inspection d'inventaire QR par drone

Implémentation de `../CONCEPTION_controleur_RL_inventaire.md`, sur **Isaac Sim 5.1 / Isaac Lab 2.3**.
Ordre des cerveaux : **PPO → Dreamer → Pore**.

## Structure

### 🟢 Cœur (le vrai code)
| Fichier | Rôle |
|---|---|
| `env.py` | **L'arène** (`QRInventoryEnv`) : entrepôt + drone vitesse + LiDAR + lecture QR + observation/récompense/fin. |
| `config_rl.py` | Tous les **paramètres** (LiDAR, action, récompense, proxy QR, scène, entraînement). Pur python. |
| `qr_task.py` | **Pose les QR** sur les cartons + donne leur **position** pour la lecture. |
| `__init__.py` | Marqueur de package. |
| `launch.sh` | **Lanceur** Isaac (écran/EGL/headless + flag pilote). |
| `assets/qr/` | Les **123 images QR** générées (une par carton). |

### 🔵 `tests/` — vérifications (une par étape)
| Fichier | Vérifie |
|---|---|
| `test_drone.py` | le drone vole selon la commande (T1.2) |
| `test_lidar.py` | le LiDAR voit l'entrepôt (T1.3) |
| `test_qr_placement.py` | pose des QR + image/fenêtre pour voir (T1.5a) |
| `test_qr_read.py` | le proxy « QR lu » (T1.5b) |

### 🗑️ `diag/` — diagnostics ponctuels (gardés pour mémoire)
`inspect_usd.py`, `find_cartons.py`, `bench_rtx_lidar.py`, `test_isaac5.py`, `test_qr_attach.py`, `render_view.py`.

## Lancer un test
```bash
# headless (sans fenêtre, recommandé)
bash launch.sh tests/test_qr_read.py --headless
# avec fenêtre (devant l'écran de la machine)
bash launch.sh tests/test_qr_placement.py
```

## Décisions clés
- LiDAR **360×5** via `MultiMeshRayCaster` (voit l'entrepôt réel, multi-mesh + dynamique).
- Contrôle **vitesse** (`write_root_velocity_to_sim`), gravité du drone coupée (altitude autoritaire).
- QR **réel par carton** (texture sur 2 faces avant/arrière) ; **proxy géométrique** (distance+FOV+angle+dwell+vitesse) à l'entraînement, vrai `pyzbar` à l'éval.
- Entrepôt chargé en **référence sur scène locale** (sinon textures non résolues).
- Flag pilote RTX obligatoire : `--kit_args="--/rtx/verifyDriverVersion/enabled=false"` (géré par `launch.sh`).
