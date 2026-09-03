"""Test de viabilité option B : Isaac Sim 5.1 + Isaac Lab 2.3 + MultiMeshRayCaster.

Confirme que : (1) Isaac Sim 5.1 démarre sur le driver 535, (2) MultiMeshRayCaster
est importable, (3) torch voit le GPU. Ne modifie rien.

Lancement (depuis le NOUVEAU venv) :
  ~/isaac5_env/bin/python ~/simulation_mc02/rl_inventory/test_isaac5.py \
      --headless --enable_cameras --kit_args="--/rtx/verifyDriverVersion/enabled=false"
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test viabilité Isaac Sim 5.1 + MultiMeshRayCaster")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

print("\n>>> [1/3] Isaac Sim 5.1 a DÉMARRÉ (app + moteur de rendu OK sur ton driver) ✅\n")

# import de MultiMeshRayCaster (plusieurs chemins possibles selon la version)
found = None
for imp in (
    "from isaaclab.sensors import MultiMeshRayCasterCfg as X",
    "from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg as X",
    "from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg as X",
):
    try:
        ns = {}
        exec(imp, ns)
        found = imp
        print(f">>> [2/3] MultiMeshRayCaster IMPORT OK ✅   ({imp})")
        break
    except Exception as e:
        print(f"    essai import échoué : {type(e).__name__}: {str(e)[:90]}")
if found is None:
    print(">>> [2/3] MultiMeshRayCaster IMPORT ÉCHEC ❌")

import torch  # noqa: E402

print(f">>> [3/3] torch {torch.__version__} | cuda_available: {torch.cuda.is_available()}")

print("\n>>> Si les 3 lignes sont ✅ → option B VIABLE (on construira le LiDAR dessus).\n")

simulation_app.close()
