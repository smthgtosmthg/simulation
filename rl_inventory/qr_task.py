"""Les QR codes des cartons : les POSER sur les cartons + donner leur POSITION pour la lecture."""

import os
import re

import qrcode
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

QR_DIR = os.path.join(os.path.dirname(__file__), "assets", "qr")
CARTON_RE = re.compile(r"SM_CardBox", re.I)
FACES = [(0, 1), (0, -1)]  # avant / arrière (axe X) — pas gauche/droite


def find_cartons(stage):
    """Trouve tous les cartons (les prims nommés SM_CardBox*) dans la scène."""
    return [p for p in stage.Traverse() if p.GetTypeName() == "Xform" and CARTON_RE.search(p.GetName())]


def _make_qr_png(data, path):
    """Génère l'image PNG d'un QR (en couleur RGB) à partir d'un texte (l'ID du carton)."""
    qrcode.make(data).save(path)
    Image.open(path).convert("RGB").save(path)


def _qr_material(stage, mat_path, png):
    """Crée le matériau qui affiche l'image QR sur une surface (noir/blanc bien contrasté)."""
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    tex = UsdShade.Shader.Define(stage, mat_path + "/DiffuseTexture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(png))
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    streader = UsdShade.Shader.Define(stage, mat_path + "/STReader")
    streader.CreateIdAttr("UsdPrimvarReader_float2")
    streader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(streader.ConnectableAPI(), "result")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _add_face_quad(stage, base_path, idx, lo, hi, axis, sign, mat):
    """Colle un petit panneau plat (avec le QR) sur UNE face du carton."""
    quad = UsdGeom.Mesh.Define(stage, f"{base_path}/QRTag_{idx}")
    a0, a1 = [i for i in range(3) if i != axis]
    c0, c1 = (lo[a0] + hi[a0]) / 2.0, (lo[a1] + hi[a1]) / 2.0
    half = 0.4 * min(hi[a0] - lo[a0], hi[a1] - lo[a1])
    coord = (hi[axis] if sign > 0 else lo[axis]) + sign * 0.01

    def pt(d0, d1):
        p = [0.0, 0.0, 0.0]
        p[axis], p[a0], p[a1] = coord, c0 + d0 * half, c1 + d1 * half
        return Gf.Vec3f(*p)

    pts = [pt(-1, -1), pt(1, -1), pt(1, 1), pt(-1, 1)] if sign > 0 else [pt(1, -1), pt(-1, -1), pt(-1, 1), pt(1, 1)]
    quad.CreatePointsAttr(pts)
    quad.CreateFaceVertexCountsAttr([4])
    quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    normal = [0.0, 0.0, 0.0]
    normal[axis] = float(sign)
    quad.CreateNormalsAttr([Gf.Vec3f(*normal)] * 4)
    quad.SetNormalsInterpolation("faceVarying")
    quad.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(quad).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
    UsdShade.MaterialBindingAPI.Apply(quad.GetPrim()).Bind(mat)


def _attach_qr(stage, prim, png):
    """Pose le QR d'un carton sur ses 2 faces (avant + arrière)."""
    base = prim.GetPath().pathString
    if stage.GetPrimAtPath(base + "/QRTag_0").IsValid():
        return
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    mat = _qr_material(stage, base + "/QRMat", png)
    for idx, (axis, sign) in enumerate(FACES):
        _add_face_quad(stage, base, idx, lo, hi, axis, sign, mat)


def carton_qr_world_poses(stage, cartons):
    """Donne la position + l'orientation (monde) de chaque QR — pour savoir si le drone le voit."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    xcache = UsdGeom.XformCache()
    positions, normals = [], []
    for prim in cartons:
        rng = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        lo, hi = rng.GetMin(), rng.GetMax()
        m = xcache.GetLocalToWorldTransform(prim)
        for axis, sign in FACES:
            others = [i for i in range(3) if i != axis]
            local = [0.0, 0.0, 0.0]
            local[axis] = hi[axis] if sign > 0 else lo[axis]
            local[others[0]] = (lo[others[0]] + hi[others[0]]) / 2.0
            local[others[1]] = (lo[others[1]] + hi[others[1]]) / 2.0
            wp = m.Transform(Gf.Vec3d(*local))
            d = [0.0, 0.0, 0.0]
            d[axis] = float(sign)
            wn = m.TransformDir(Gf.Vec3d(*d)).GetNormalized()
            positions.append([wp[0], wp[1], wp[2]])
            normals.append([wn[0], wn[1], wn[2]])
    return positions, normals


def attach_qr_to_cartons(stage, limit=None):
    """Pose un QR unique sur CHAQUE carton de l'entrepôt (génère l'image + colle les panneaux)."""
    os.makedirs(QR_DIR, exist_ok=True)
    cartons = find_cartons(stage)
    if limit:
        cartons = cartons[:limit]
    info = {}
    for i, prim in enumerate(cartons):
        box_id = f"CARTON_{i:04d}"
        png = os.path.join(QR_DIR, f"{box_id}.png")
        _make_qr_png(box_id, png)
        _attach_qr(stage, prim, png)
        info[box_id] = prim.GetPath().pathString
    return info
