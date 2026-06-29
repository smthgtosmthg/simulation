"""Capture quelques images RGB de l'entrepôt (pour visualiser cartons vs bacs KLT)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import os

import numpy as np
import omni.replicator.core as rep
import omni.usd

USD = os.getenv(
    "AIF_FACTORY_USD",
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
)
OUT = "/home/djihene_guitoun/simulation_mc02/rl_inventory"

omni.usd.get_context().open_stage(USD)
for _ in range(30):
    simulation_app.update()

views = [
    ((-10.0, -10.0, 7.0), (2.0, 6.0, 1.5), "overview"),
    ((3.0, 11.0, 2.2), (9.0, 11.0, 1.4), "cardbox_closeup"),
    ((-2.0, 2.0, 2.0), (-10.0, 8.0, 1.5), "aisle"),
]

for pos, tgt, name in views:
    cam = rep.create.camera(position=pos, look_at=tgt)
    rp = rep.create.render_product(cam, (1280, 720))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])
    for _ in range(45):
        simulation_app.update()
    data = np.asarray(rgb.get_data())
    path = f"{OUT}/view_{name}.png"
    try:
        from PIL import Image

        Image.fromarray(data[..., :3]).save(path)
        print(f"SAVED {path}  shape={data.shape}")
    except Exception as e:
        np.save(path.replace(".png", ".npy"), data)
        print(f"PIL KO ({e}) -> {path.replace('.png', '.npy')}")
    rgb.detach([rp])
    rp.destroy()

simulation_app.close()
