"""Liste les objets/cartons de l'entrepôt USD (lecture seule, ne modifie rien)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import os
import re
from collections import Counter

import omni.usd
from pxr import Usd, UsdGeom

USD = os.getenv(
    "AIF_FACTORY_USD",
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
)

omni.usd.get_context().open_stage(USD)
stage = omni.usd.get_context().get_stage()
prims = list(stage.Traverse())


def base_name(name):
    return re.sub(r"(_\d+)+$", "", name)


catalog = Counter(base_name(p.GetName()) for p in prims if p.IsA(UsdGeom.Xformable))

print("\n=== CATALOGUE (noms normalisés, top 30) ===")
for name, n in catalog.most_common(30):
    print(f"  {n:>5}  {name}")

pattern = re.compile(r"klt|cardbox|carton|crate|bin|_box", re.I)
cartons = [p for p in prims if pattern.search(p.GetName())]
print(f"\n=== Prims 'carton' candidats (KLT/Cardbox/Carton/Box/Bin/Crate) : {len(cartons)} ===")

cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
print("\n=== Échantillon (8 cartons) : chemin | centre monde (m) | taille (m) ===")
for p in cartons[:8]:
    try:
        rng = cache.ComputeWorldBound(p).ComputeAlignedRange()
        lo, hi = rng.GetMin(), rng.GetMax()
        c = [(lo[i] + hi[i]) / 2 for i in range(3)]
        s = [hi[i] - lo[i] for i in range(3)]
        print(f"  {p.GetPath()}")
        print(f"      type={p.GetTypeName()}  centre=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})  taille=({s[0]:.2f},{s[1]:.2f},{s[2]:.2f})")
    except Exception as e:
        print(f"  {p.GetPath()}  bbox err: {e}")

simulation_app.close()
