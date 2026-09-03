"""Génération des images de QR et collage sur les faces des cartons.

La charge utile fait exactement 7 caractères pour tenir en version 1 (21 modules) avec la
correction d'erreur la plus forte. Un identifiant plus long ferait passer en version 2, soit
25 modules, donc des modules 16 % plus fins et plus durs à lire de loin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import qrcode
from qrcode.constants import ERROR_CORRECT_H

from .config import QR_CFG

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "qr"


@dataclass(frozen=True)
class TagPose:
    tag_id: str
    box_path: str
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    size: float


def payload(index: int) -> str:
    return QR_CFG.payload_fmt.format(index)


def generate_images(count: int, out_dir: Path = ASSET_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    made = {}
    for i in range(count):
        data = payload(i)
        qr = qrcode.QRCode(version=1, error_correction=ERROR_CORRECT_H, box_size=20, border=2)
        qr.add_data(data)
        qr.make(fit=True)
        if qr.version != 1:
            raise ValueError(f"{data!r} sort en version {qr.version}, la charge utile est trop longue")
        path = out_dir / f"{data}.png"
        qr.make_image(fill_color="black", back_color="white").convert("RGB").save(path, "PNG")
        made[data] = path
    return made


def _material(stage, path: str, png: Path):
    from pxr import Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    reader = UsdShade.Shader.Define(stage, f"{path}/ST")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    tex = UsdShade.Shader.Define(stage, f"{path}/Tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(png)))
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _local_bounds(prim):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
    return rng.GetMin(), rng.GetMax()


def _add_panel(stage, box_path: str, idx: int, lo, hi, axis: int, sign: int, mat):
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    a0, a1 = [i for i in range(3) if i != axis]
    c0, c1 = (lo[a0] + hi[a0]) / 2.0, (lo[a1] + hi[a1]) / 2.0
    half = QR_CFG.panel_ratio * min(hi[a0] - lo[a0], hi[a1] - lo[a1])
    coord = (hi[axis] if sign > 0 else lo[axis]) + sign * QR_CFG.offset

    def pt(d0, d1):
        p = [0.0, 0.0, 0.0]
        p[axis], p[a0], p[a1] = coord, c0 + d0 * half, c1 + d1 * half
        return Gf.Vec3f(*p)

    quad = UsdGeom.Mesh.Define(stage, f"{box_path}/QRTag_{idx}")
    pts = [pt(-1, -1), pt(1, -1), pt(1, 1), pt(-1, 1)]
    if sign < 0:
        pts = [pt(1, -1), pt(-1, -1), pt(-1, 1), pt(1, 1)]
    quad.CreatePointsAttr(pts)
    quad.CreateFaceVertexCountsAttr([4])
    quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    normal = [0.0, 0.0, 0.0]
    normal[axis] = float(sign)
    quad.CreateNormalsAttr([Gf.Vec3f(*normal)] * 4)
    quad.SetNormalsInterpolation("faceVarying")
    quad.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(quad).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    ).Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
    UsdShade.MaterialBindingAPI.Apply(quad.GetPrim()).Bind(mat)
    return 2.0 * half


def attach(stage, box_paths: list[str], images: dict[str, Path]) -> list[TagPose]:
    """Colle un QR sur chaque face déclarée des cartons et renvoie la pose réelle des panneaux."""
    from pxr import Gf, Usd, UsdGeom

    xcache = UsdGeom.XformCache()
    poses: list[TagPose] = []

    for i, box_path in enumerate(sorted(box_paths)):
        prim = stage.GetPrimAtPath(box_path)
        if not prim or not prim.IsValid():
            continue
        tag_id = payload(i)
        png = images.get(tag_id)
        if png is None:
            continue
        lo, hi = _local_bounds(prim)
        if (hi - lo).GetLength() <= 0.0:
            continue
        mat = _material(stage, f"{box_path}/QRMat", png)
        m = xcache.GetLocalToWorldTransform(prim)

        for idx, (axis, sign) in enumerate(QR_CFG.faces):
            size = _add_panel(stage, box_path, idx, lo, hi, axis, sign, mat)
            local = [0.0, 0.0, 0.0]
            a0, a1 = [k for k in range(3) if k != axis]
            local[axis] = (hi[axis] if sign > 0 else lo[axis]) + sign * QR_CFG.offset
            local[a0] = (lo[a0] + hi[a0]) / 2.0
            local[a1] = (lo[a1] + hi[a1]) / 2.0
            direction = [0.0, 0.0, 0.0]
            direction[axis] = float(sign)
            pos = m.Transform(Gf.Vec3d(*local))
            nrm = m.TransformDir(Gf.Vec3d(*direction)).GetNormalized()
            poses.append(
                TagPose(
                    tag_id=tag_id,
                    box_path=box_path,
                    position=(pos[0], pos[1], pos[2]),
                    normal=(nrm[0], nrm[1], nrm[2]),
                    size=size,
                )
            )
    return poses
