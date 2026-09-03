"""T1.5a (sans rendu) — pose les QR sur les cartons et vérifie."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import os
import sys

import omni.usd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rl_inventory.qr_task import QR_DIR, attach_qr_to_cartons  # noqa: E402

USD = os.getenv(
    "AIF_FACTORY_USD",
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
)

omni.usd.get_context().open_stage(USD)
stage = omni.usd.get_context().get_stage()
info = attach_qr_to_cartons(stage)

print(f"\nQR posés sur {len(info)} cartons")
for box_id, path in list(info.items())[:3]:
    tag = stage.GetPrimAtPath(path + "/QRTag")
    print(f"  {box_id}: {path}/QRTag  valide={tag.IsValid()} type={tag.GetTypeName()}")
n_png = len([f for f in os.listdir(QR_DIR) if f.endswith(".png")]) if os.path.isdir(QR_DIR) else 0
print(f"PNG QR générés : {n_png} dans {QR_DIR}")

simulation_app.close()
