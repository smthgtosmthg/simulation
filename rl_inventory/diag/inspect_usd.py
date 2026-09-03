"""Inspection LECTURE SEULE de l'entrepôt USD.

Ne modifie AUCUN fichier et ne touche PAS à l'environnement. Sert uniquement à
voir comment l'entrepôt USD est construit (nombre de meshes, leur taille,
dimensions) pour choisir la meilleure façon de donner un LiDAR au drone dans CE
vrai environnement.

Lancement :
  cd ~/IsaacLab
  ./isaaclab.sh -p ~/simulation_mc02/rl_inventory/inspect_usd.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspection lecture seule de l'entrepôt USD")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import os  # noqa: E402

import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

USD = os.getenv(
    "AIF_FACTORY_USD",
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
)

print(f"\n[inspect] Ouverture (lecture seule) de : {USD}\n")
ctx = omni.usd.get_context()
ctx.open_stage(USD)
stage = ctx.get_stage()

prims = list(stage.Traverse())
meshes = [p for p in prims if p.GetTypeName() == "Mesh"]
xforms = [p for p in prims if p.GetTypeName() == "Xform"]
instanceable = [p for p in prims if p.IsInstanceable()]

print("================= STRUCTURE USD =================")
print(f"prims totaux : {len(prims)}")
print(f"Mesh         : {len(meshes)}")
print(f"Xform        : {len(xforms)}")
print(f"instanceable : {len(instanceable)}")

sizes = []
for m in meshes:
    pts = UsdGeom.Mesh(m).GetPointsAttr().Get()
    sizes.append((len(pts) if pts else 0, str(m.GetPath())))
sizes.sort(reverse=True)

print("\n--- 15 plus gros meshes (nb de points) ---")
for n, path in sizes[:15]:
    print(f"  {n:>8} pts   {path}")

total_pts = sum(n for n, _ in sizes)
print(f"\npoints totaux (tous meshes) : {total_pts}")
if sizes and total_pts > 0:
    print(f"plus gros mesh = {sizes[0][0]} pts ({100 * sizes[0][0] / total_pts:.1f}% du total)")

try:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    print(f"\nbounding box monde : X[{lo[0]:.1f},{hi[0]:.1f}] Y[{lo[1]:.1f},{hi[1]:.1f}] Z[{lo[2]:.1f},{hi[2]:.1f}]")
    print(f"taille entrepôt    : {hi[0] - lo[0]:.1f} x {hi[1] - lo[1]:.1f} x {hi[2] - lo[2]:.1f} m")
except Exception as e:
    print("bbox: erreur", e)

print("================================================\n")

simulation_app.close()
