"""T1.5a — pose les QR sur les cartons.

- avec --headless : rend une image (view_qr_closeup.png).
- sans --headless : ouvre la fenêtre Isaac Sim pour naviguer (ferme la fenêtre pour quitter).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import os
import sys

import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rl_inventory.qr_task import attach_qr_to_cartons  # noqa: E402

USD = os.getenv(
    "AIF_FACTORY_USD",
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
)
OUT = "/home/djihene_guitoun/simulation_mc02/rl_inventory"


def pump(n):
    for _ in range(n):
        simulation_app.update()


try:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(USD, "/World/Warehouse")
    pump(150)

    info = attach_qr_to_cartons(stage)
    print(f"QR posés sur {len(info)} cartons")
    pump(30)

    if args.headless:
        cam = rep.create.camera(position=(5.5, 11.0, 1.7), look_at=(9.0, 11.0, 1.5))
        rp = rep.create.render_product(cam, (1280, 720))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([rp])
        pump(60)
        Image.fromarray(np.asarray(rgb.get_data())[..., :3]).save(f"{OUT}/view_qr_closeup.png")
        print(f"SAVED {OUT}/view_qr_closeup.png")
    else:
        try:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(eye=[5.5, 11.0, 1.7], target=[9.0, 11.0, 1.5])
        except Exception as e:
            print(f"(caméra non placée: {e})")
        print(">>> Prêt. Clic DROIT maintenu + WASD pour voler. Ferme la fenêtre pour quitter.")
        while simulation_app.is_running():
            simulation_app.update()
finally:
    simulation_app.close()
