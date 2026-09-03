#!/usr/bin/env python3
"""Generate Sionna/Mitsuba warehouse scene (XML + OBJ meshes)
matching the Gazebo warehouse_drones.sdf geometry exactly."""

import os

BASE = os.path.expanduser(
    "~/ns-allinone-3.40/ns-3.40/contrib/sionna/model/ns3sionna/models/warehouse"
)
MESH_DIR = os.path.join(BASE, "meshes")
os.makedirs(MESH_DIR, exist_ok=True)


# ── Helper: write a box as OBJ faces ──────────────────────────────
def box_obj(cx, cy, cz, sx, sy, sz, v_offset=0):
    """Return (vertices_str, faces_str, n_verts=8) for a box."""
    x1, x2 = cx - sx / 2, cx + sx / 2
    y1, y2 = cy - sy / 2, cy + sy / 2
    z1, z2 = cz - sz / 2, cz + sz / 2

    verts = [
        (x1, y1, z1),  # 1
        (x2, y1, z1),  # 2
        (x2, y2, z1),  # 3
        (x1, y2, z1),  # 4
        (x1, y1, z2),  # 5
        (x2, y1, z2),  # 6
        (x2, y2, z2),  # 7
        (x1, y2, z2),  # 8
    ]
    o = v_offset  # vertex index offset (0-based)
    faces = [
        (4+o, 3+o, 2+o, 1+o),  # bottom  -Z
        (5+o, 6+o, 7+o, 8+o),  # top     +Z
        (1+o, 2+o, 6+o, 5+o),  # front   -Y
        (3+o, 4+o, 8+o, 7+o),  # back    +Y
        (4+o, 1+o, 5+o, 8+o),  # left    -X
        (2+o, 3+o, 7+o, 6+o),  # right   +X
    ]
    v_str = "\n".join(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}" for v in verts)
    f_str = "\n".join(f"f {f[0]} {f[1]} {f[2]} {f[3]}" for f in faces)
    return v_str, f_str, 8


def quad_obj(v1, v2, v3, v4, v_offset=0):
    """Single quad face."""
    o = v_offset
    v_str = "\n".join(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}" for v in [v1, v2, v3, v4])
    f_str = f"f {1+o} {2+o} {3+o} {4+o}"
    return v_str, f_str, 4


# ══════════════════════════════════════════════════════════════════
# 1. FLOOR  (30×20 at z=0)
# ══════════════════════════════════════════════════════════════════
with open(os.path.join(MESH_DIR, "floor.obj"), "w") as f:
    f.write("# Warehouse floor (30m x 20m)\no floor\n")
    v, face, _ = quad_obj((-15, -10, 0), (15, -10, 0), (15, 10, 0), (-15, 10, 0))
    f.write(v + "\nvn 0 0 1\n")
    f.write("f 1//1 2//1 3//1 4//1\n")
print("  floor.obj")

# ══════════════════════════════════════════════════════════════════
# 2. CEILING  (30×20 at z=6)
# ══════════════════════════════════════════════════════════════════
with open(os.path.join(MESH_DIR, "ceiling.obj"), "w") as f:
    f.write("# Warehouse ceiling (30m x 20m at z=6)\no ceiling\n")
    # face normal pointing down (inward)
    v, face, _ = quad_obj((-15, -10, 6), (15, -10, 6), (15, 10, 6), (-15, 10, 6))
    f.write(v + "\nvn 0 0 -1\n")
    f.write("f 4//1 3//1 2//1 1//1\n")
print("  ceiling.obj")

# ══════════════════════════════════════════════════════════════════
# 3. WALLS  (5 wall panels as quads — using twosided BSDF)
# ══════════════════════════════════════════════════════════════════
wall_quads = [
    # North wall: y=10, full width
    [(-15, 10, 0), (15, 10, 0), (15, 10, 6), (-15, 10, 6)],
    # South wall left: y=-10, x from -15 to -6
    [(-15, -10, 0), (-6, -10, 0), (-6, -10, 6), (-15, -10, 6)],
    # South wall right: y=-10, x from 6 to 15
    [(6, -10, 0), (15, -10, 0), (15, -10, 6), (6, -10, 6)],
    # East wall: x=15
    [(15, -10, 0), (15, 10, 0), (15, 10, 6), (15, -10, 6)],
    # West wall: x=-15
    [(-15, 10, 0), (-15, -10, 0), (-15, -10, 6), (-15, 10, 6)],
]

with open(os.path.join(MESH_DIR, "walls.obj"), "w") as f:
    f.write("# Warehouse walls\no walls\n")
    vi = 0
    for q in wall_quads:
        for v in q:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        a, b, c, d = vi + 1, vi + 2, vi + 3, vi + 4
        f.write(f"f {a} {b} {c} {d}\n")
        vi += 4
print("  walls.obj")

# ══════════════════════════════════════════════════════════════════
# 4. SHELVES  (4 shelf boxes, each 6×1.2×3m, centered at z=1.5)
# ══════════════════════════════════════════════════════════════════
shelves = [
    (-7,  5, 1.5, 6, 1.2, 3),   # NW
    ( 7,  5, 1.5, 6, 1.2, 3),   # NE
    (-7, -5, 1.5, 6, 1.2, 3),   # SW
    ( 7, -5, 1.5, 6, 1.2, 3),   # SE
]

with open(os.path.join(MESH_DIR, "shelves.obj"), "w") as f:
    f.write("# Warehouse shelves (4 units)\no shelves\n")
    vi = 0
    for cx, cy, cz, sx, sy, sz in shelves:
        v_str, f_str, nv = box_obj(cx, cy, cz, sx, sy, sz, v_offset=vi)
        f.write(v_str + "\n" + f_str + "\n")
        vi += nv
print("  shelves.obj")

# ══════════════════════════════════════════════════════════════════
# 5. CRATES  (2 crates)
# ══════════════════════════════════════════════════════════════════
crates = [
    (-12, -7, 0.5, 1.0, 1.0, 1.0),
    ( 12,  7, 0.4, 1.2, 0.8, 0.8),
]

with open(os.path.join(MESH_DIR, "crates.obj"), "w") as f:
    f.write("# Warehouse crates\no crates\n")
    vi = 0
    for cx, cy, cz, sx, sy, sz in crates:
        v_str, f_str, nv = box_obj(cx, cy, cz, sx, sy, sz, v_offset=vi)
        f.write(v_str + "\n" + f_str + "\n")
        vi += nv
print("  crates.obj")


# ══════════════════════════════════════════════════════════════════
# 6. XML SCENE FILE (Mitsuba 2.1 format for Sionna)
# ══════════════════════════════════════════════════════════════════
xml = """\
<?xml version="1.0"?>
<!--
  Warehouse scene for Sionna ray-tracing.
  Matches the Gazebo warehouse_drones.sdf geometry:
    30m x 20m footprint, 6m ceiling height,
    4 metal shelves (6x1.2x3m), 2 crates.
-->
<scene version="2.1.0">

<!-- Rendering -->
    <integrator type="path">
        <integer name="max_depth" value="12"/>
    </integrator>

<!-- ════════ Materials ════════ -->

    <!-- Concrete: floor, ceiling, walls -->
    <bsdf type="twosided" id="mat-itu_concrete">
        <bsdf type="diffuse">
            <rgb value="0.5 0.5 0.5" name="reflectance"/>
        </bsdf>
    </bsdf>

    <!-- Metal: warehouse shelving -->
    <bsdf type="twosided" id="mat-itu_metal">
        <bsdf type="diffuse">
            <rgb value="0.6 0.6 0.65" name="reflectance"/>
        </bsdf>
    </bsdf>

    <!-- Wood: crates -->
    <bsdf type="twosided" id="mat-itu_wood">
        <bsdf type="diffuse">
            <rgb value="0.65 0.45 0.2" name="reflectance"/>
        </bsdf>
    </bsdf>

<!-- ════════ Emitters ════════ -->

    <emitter type="constant" id="World">
        <rgb value="1.0 1.0 1.0" name="radiance"/>
    </emitter>

<!-- ════════ Shapes ════════ -->

    <shape type="obj" id="mesh-floor">
        <string name="filename" value="meshes/floor.obj"/>
        <boolean name="face_normals" value="true"/>
        <ref id="mat-itu_concrete" name="bsdf"/>
    </shape>

    <shape type="obj" id="mesh-ceiling">
        <string name="filename" value="meshes/ceiling.obj"/>
        <boolean name="face_normals" value="true"/>
        <ref id="mat-itu_concrete" name="bsdf"/>
    </shape>

    <shape type="obj" id="mesh-walls">
        <string name="filename" value="meshes/walls.obj"/>
        <boolean name="face_normals" value="true"/>
        <ref id="mat-itu_concrete" name="bsdf"/>
    </shape>

    <shape type="obj" id="mesh-shelves">
        <string name="filename" value="meshes/shelves.obj"/>
        <boolean name="face_normals" value="true"/>
        <ref id="mat-itu_metal" name="bsdf"/>
    </shape>

    <shape type="obj" id="mesh-crates">
        <string name="filename" value="meshes/crates.obj"/>
        <boolean name="face_normals" value="true"/>
        <ref id="mat-itu_wood" name="bsdf"/>
    </shape>

</scene>
"""

xml_path = os.path.join(BASE, "warehouse.xml")
with open(xml_path, "w") as f:
    f.write(xml)
print(f"  warehouse.xml")

print(f"\n  All files written to:\n  {BASE}/")
print(f"  ├─ warehouse.xml")
print(f"  └─ meshes/")
print(f"     ├─ floor.obj")
print(f"     ├─ ceiling.obj")
print(f"     ├─ walls.obj")
print(f"     ├─ shelves.obj")
print(f"     └─ crates.obj")
