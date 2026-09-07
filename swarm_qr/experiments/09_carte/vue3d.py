"""La carte en trois dimensions : un fichier HTML autonome que n'importe quel navigateur ouvre.

On y tourne autour de l'entrepôt à la souris. Les cubes sombres sont les obstacles vus par le
lidar, les cubes bleutés ce qui a été vu d'assez près pour lire, les billes vertes les codes
lus, les billes orange les motifs repérés sans être lus, le trait bleu la trajectoire du
drone. Les racks vrais s'affichent en fil de fer rouge pour comparer la carte à la vérité.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from swarm_qr import mapping  # noqa: E402
from swarm_qr.env.config import INTERIOR  # noqa: E402

TRAJ_MAX = 4000


def donnees(carte: mapping.Carte, vol: dict) -> dict:
    occ = np.argwhere(carte.occupation > mapping.SEUIL_OCCUPE)
    couv = np.argwhere(carte.couverture > 0)
    traj = vol.get("trajectoire", [])
    if len(traj) > TRAJ_MAX:
        pas = len(traj) // TRAJ_MAX + 1
        traj = traj[::pas]
    return {
        "cell": carte.g.cell,
        "origine": carte.origine.tolist(),
        "forme": list(carte.forme),
        "interieur": [INTERIOR.x_min, INTERIOR.x_max, INTERIOR.y_min, INTERIOR.y_max, 0.0,
                      INTERIOR.z_ceiling],
        "occupes": occ.astype(int).ravel().tolist(),
        "couverts": couv.astype(int).ravel().tolist(),
        "panneaux": [{"c": p.code, "p": [round(float(x), 3) for x in p.position], "n": p.lectures}
                     for p in carte.panneaux],
        "pistes": [[round(float(x), 3) for x in q.position] for q in carte.pistes],
        "trajectoire": [[round(float(x), 3) for x in t] for t in traj],
        "racks": vol.get("racks", []),
        "verite": [t["position"] for t in vol.get("verite", [])],
        "resume": carte.resume(),
    }


GABARIT = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Carte 3D — entrepôt découvert</title>
<style>
  html, body { margin: 0; height: 100%; background: #0b1020; color: #e6edf3;
               font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; overflow: hidden; }
  #scene { position: absolute; inset: 0; }
  #panneau { position: absolute; top: 14px; left: 14px; width: 250px; padding: 12px 14px;
             background: rgba(13, 20, 38, .86); border: 1px solid rgba(255,255,255,.08);
             border-radius: 10px; backdrop-filter: blur(6px); }
  #panneau h1 { font-size: 15px; margin: 0 0 6px; font-weight: 600; }
  #panneau p { margin: 0 0 10px; color: #9fb0c8; }
  label { display: flex; align-items: center; gap: 8px; margin: 5px 0; cursor: pointer; }
  .pastille { width: 12px; height: 12px; border-radius: 3px; flex: none; }
  .rond { border-radius: 50%; }
  #chiffres { margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,.08);
              color: #c9d4e3; }
  #chiffres b { color: #fff; }
  #bulle { position: absolute; display: none; padding: 5px 8px; background: #111827;
           border: 1px solid #22c55e; border-radius: 6px; pointer-events: none; font-size: 12px; }
  #temps { position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
           width: min(600px, 80vw); display: flex; gap: 10px; align-items: center;
           background: rgba(13, 20, 38, .86); padding: 8px 12px; border-radius: 10px;
           border: 1px solid rgba(255,255,255,.08); }
  #temps input { flex: 1; }
  #temps button { background: #2563eb; color: #fff; border: 0; border-radius: 6px;
                  padding: 5px 12px; cursor: pointer; }
  #aide { position: absolute; right: 14px; bottom: 16px; color: #7c8aa3; font-size: 12px;
          text-align: right; }
</style>
</head>
<body>
<div id="scene"></div>
<div id="panneau">
  <h1>Carte découverte par le drone</h1>
  <p>Tout ce qui est affiché vient des capteurs, sauf la vérité en rouge.</p>
  <label><span class="pastille" style="background:#4b5563"></span><input type="checkbox" id="c_occ" checked> obstacles vus par le lidar</label>
  <label><span class="pastille" style="background:#7dd3fc"></span><input type="checkbox" id="c_couv" checked> vu d'assez près pour lire</label>
  <label><span class="pastille rond" style="background:#22c55e"></span><input type="checkbox" id="c_lus" checked> codes lus</label>
  <label><span class="pastille rond" style="background:#f59e0b"></span><input type="checkbox" id="c_pistes" checked> repérés, non lus</label>
  <label><span class="pastille" style="background:#60a5fa"></span><input type="checkbox" id="c_traj" checked> trajectoire du drone</label>
  <label><span class="pastille" style="background:#ef4444; opacity:.8"></span><input type="checkbox" id="c_verite"> vérité : racks et panneaux</label>
  <div id="chiffres"></div>
</div>
<div id="bulle"></div>
<div id="temps"><button id="lecture">Rejouer</button><input type="range" id="curseur" min="0" max="1000" value="1000"><span id="horloge">0 s</span></div>
<div id="aide">glisser : tourner · molette : zoom · clic droit : déplacer · survoler une bille verte : le code</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script>
const D = __DONNEES__;
const cell = D.cell, O = D.origine;
const centre = i => [O[0] + (i[0] + .5) * cell, O[1] + (i[1] + .5) * cell, O[2] + (i[2] + .5) * cell];

const conteneur = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
conteneur.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x0b1020, 45, 90);
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 300);
camera.up.set(0, 0, 1);

const [x0, x1, y0, y1, z0, z1] = D.interieur;
const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
let cible = new THREE.Vector3(cx, cy, 1.5);
let theta = -0.9, phi = 0.9, rayon = 34;
function placeCamera() {
  camera.position.set(cible.x + rayon * Math.cos(phi) * Math.cos(theta),
                      cible.y + rayon * Math.cos(phi) * Math.sin(theta),
                      cible.z + rayon * Math.sin(phi));
  camera.lookAt(cible);
}
placeCamera();

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const soleil = new THREE.DirectionalLight(0xffffff, 0.8);
soleil.position.set(-20, -30, 40);
scene.add(soleil);
const contre = new THREE.DirectionalLight(0x88aaff, 0.35);
contre.position.set(30, 20, 10);
scene.add(contre);

// le sol et les murs de l'entrepôt
const grille = new THREE.GridHelper(Math.max(x1 - x0, y1 - y0) + 4, 24, 0x334155, 0x1f2937);
grille.rotation.x = Math.PI / 2; grille.position.set(cx, cy, 0.0);
scene.add(grille);
const murs = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(x1 - x0, y1 - y0, z1 - z0)),
  new THREE.LineBasicMaterial({ color: 0x475569 }));
murs.position.set(cx, cy, (z0 + z1) / 2);
scene.add(murs);

// les cubes de la carte, en un seul maillage instancié par couche
function cubes(indices, couleur, opacite, taille) {
  const n = indices.length / 3;
  const geo = new THREE.BoxGeometry(cell * taille, cell * taille, cell * taille);
  const mat = new THREE.MeshLambertMaterial({ color: couleur, transparent: opacite < 1,
                                             opacity: opacite, depthWrite: opacite >= 1 });
  const m = new THREE.InstancedMesh(geo, mat, n);
  const M = new THREE.Matrix4();
  for (let k = 0; k < n; k++) {
    const c = centre([indices[3 * k], indices[3 * k + 1], indices[3 * k + 2]]);
    M.makeTranslation(c[0], c[1], c[2]);
    m.setMatrixAt(k, M);
  }
  m.instanceMatrix.needsUpdate = true;
  return m;
}
const occ = cubes(D.occupes, 0x4b5563, 1.0, 0.96);
const couv = cubes(D.couverts, 0x7dd3fc, 0.10, 0.72);
scene.add(occ); scene.add(couv);

// les panneaux et les pistes
const groupeLus = new THREE.Group(), groupePistes = new THREE.Group();
const geoBille = new THREE.SphereGeometry(0.21, 16, 12);
const matLu = new THREE.MeshStandardMaterial({ color: 0x22c55e, emissive: 0x166534, emissiveIntensity: .9, roughness: .35 });
const matPiste = new THREE.MeshStandardMaterial({ color: 0xf59e0b, emissive: 0x7c2d12, roughness: .5 });
const billes = [];
for (const p of D.panneaux) {
  const b = new THREE.Mesh(geoBille, matLu); b.position.set(...p.p); b.userData = p;
  groupeLus.add(b); billes.push(b);
}
for (const q of D.pistes) {
  const b = new THREE.Mesh(new THREE.SphereGeometry(0.11, 10, 8), matPiste); b.position.set(...q);
  groupePistes.add(b);
}
scene.add(groupeLus); scene.add(groupePistes);

// la trajectoire et le drone
const T = D.trajectoire;
const pts = T.map(t => new THREE.Vector3(t[1], t[2], t[3]));
const traj = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
                            new THREE.LineBasicMaterial({ color: 0x60a5fa }));
scene.add(traj);
const drone = new THREE.Mesh(new THREE.SphereGeometry(0.28, 16, 12),
  new THREE.MeshStandardMaterial({ color: 0x3b82f6, emissive: 0x1e3a8a }));
if (pts.length) drone.position.copy(pts[pts.length - 1]);
scene.add(drone);

// la vérité, pour comparer
const verite = new THREE.Group();
for (const r of D.racks) {
  const b = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(r.x[1] - r.x[0], r.y[1] - r.y[0], 6.0)),
    new THREE.LineBasicMaterial({ color: 0xef4444 }));
  b.position.set((r.x[0] + r.x[1]) / 2, (r.y[0] + r.y[1]) / 2, 3.0);
  verite.add(b);
}
const matVrai = new THREE.MeshBasicMaterial({ color: 0xef4444, wireframe: true });
for (const p of D.verite) {
  const b = new THREE.Mesh(new THREE.SphereGeometry(0.12, 8, 6), matVrai); b.position.set(...p);
  verite.add(b);
}
verite.visible = false;
scene.add(verite);

// les chiffres
const R = D.resume;
document.getElementById('chiffres').innerHTML =
  `<b>${(R.part_connue * 100).toFixed(1)} %</b> de l'entrepôt connu<br>` +
  `<b>${R.occupees.toLocaleString('fr')}</b> cubes occupés · <b>${R.couvertes.toLocaleString('fr')}</b> cubes lisibles<br>` +
  `<b>${R.codes_lus}</b> codes lus sur <b>${R.faces_lues}</b> faces · <b>${R.pistes}</b> pistes<br>` +
  `${(R.octets / 1e6).toFixed(1)} Mo de carte`;

// les cases à cocher
const lie = (id, objet) => document.getElementById(id).addEventListener('change', e => objet.visible = e.target.checked);
lie('c_occ', occ); lie('c_couv', couv); lie('c_lus', groupeLus); lie('c_pistes', groupePistes);
lie('c_traj', traj); lie('c_verite', verite);

// la souris : tourner, zoomer, déplacer
let bouton = -1, dernier = null;
renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());
renderer.domElement.addEventListener('pointerdown', e => { bouton = e.button; dernier = [e.clientX, e.clientY]; });
addEventListener('pointerup', () => bouton = -1);
addEventListener('pointermove', e => {
  survol(e);
  if (bouton < 0) return;
  const dx = e.clientX - dernier[0], dy = e.clientY - dernier[1]; dernier = [e.clientX, e.clientY];
  if (bouton === 0 && !e.shiftKey) { theta -= dx * 0.006; phi = Math.min(1.5, Math.max(0.05, phi + dy * 0.006)); }
  else {
    const droite = new THREE.Vector3().crossVectors(camera.getWorldDirection(new THREE.Vector3()), camera.up).normalize();
    const haut = new THREE.Vector3().crossVectors(droite, camera.getWorldDirection(new THREE.Vector3())).normalize();
    cible.addScaledVector(droite, -dx * rayon * 0.0012).addScaledVector(haut, dy * rayon * 0.0012);
  }
  placeCamera();
});
renderer.domElement.addEventListener('wheel', e => { rayon = Math.min(120, Math.max(3, rayon * (1 + e.deltaY * 0.001))); placeCamera(); }, { passive: true });
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });

// le survol d'un code lu
const rayo = new THREE.Raycaster(), bulle = document.getElementById('bulle');
function survol(e) {
  const v = new THREE.Vector2(e.clientX / innerWidth * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  rayo.setFromCamera(v, camera);
  const hit = groupeLus.visible ? rayo.intersectObjects(billes)[0] : null;
  if (hit) {
    const p = hit.object.userData;
    bulle.style.display = 'block'; bulle.style.left = (e.clientX + 12) + 'px'; bulle.style.top = (e.clientY + 12) + 'px';
    bulle.textContent = `${p.c} — ${p.n} lecture${p.n > 1 ? 's' : ''} — (${p.p.map(x => x.toFixed(2)).join(', ')})`;
  } else bulle.style.display = 'none';
}

// rejouer le vol
const curseur = document.getElementById('curseur'), horloge = document.getElementById('horloge');
let lecture = false, avance = 1000;
function montre(k) {
  const n = Math.max(1, Math.round(k / 1000 * (pts.length - 1)));
  traj.geometry.setDrawRange(0, n + 1);
  drone.position.copy(pts[Math.min(n, pts.length - 1)]);
  horloge.textContent = T.length ? `${T[Math.min(n, T.length - 1)][0].toFixed(0)} s` : '';
}
curseur.addEventListener('input', e => { avance = +e.target.value; lecture = false; montre(avance); });
document.getElementById('lecture').addEventListener('click', () => { if (avance >= 1000) avance = 0; lecture = !lecture; });
if (pts.length) montre(1000);

function anime() {
  requestAnimationFrame(anime);
  if (lecture) { avance = Math.min(1000, avance + 1.5); curseur.value = avance; montre(avance); if (avance >= 1000) lecture = false; }
  renderer.render(scene, camera);
}
anime();
</script>
</body>
</html>
"""


def image(carte: mapping.Carte, vol: dict, sortie: Path) -> Path:
    """La même carte en image fixe, pour la fiche et le mémoire : deux points de vue."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = donnees(carte, vol)
    occ = np.array(d["occupes"]).reshape(-1, 3)
    couv = np.array(d["couverts"]).reshape(-1, 3)
    c = lambda idx: carte.centre(idx) if len(idx) else np.zeros((0, 3))
    x0, x1, y0, y1, _, _ = d["interieur"]
    # seulement la bande de vol, à l'intérieur des murs : les impacts sur le sol, le plafond et
    # les murs formeraient des nappes de points qui cachent les racks
    dedans = lambda P: P[(P[:, 2] > 0.6) & (P[:, 2] < 5.5) & (P[:, 0] > x0 + 0.3) & (P[:, 0] < x1 - 0.3)
                         & (P[:, 1] > y0 + 0.3) & (P[:, 1] < y1 - 0.3)]
    P_occ, P_couv = dedans(c(occ)), dedans(c(couv[::2]))
    traj = np.array([t[1:] for t in d["trajectoire"]]) if d["trajectoire"] else np.zeros((0, 3))
    lus = np.array([p["p"] for p in d["panneaux"]]) if d["panneaux"] else np.zeros((0, 3))
    pistes = np.array(d["pistes"]) if d["pistes"] else np.zeros((0, 3))

    fig = plt.figure(figsize=(16, 8), facecolor="#0b1020")
    for k, (elev, azim, titre) in enumerate(((50, -60, "vue d'ensemble"), (24, -135, "depuis le sud-est"))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d", facecolor="#0b1020")
        for z in (0.0, 6.0):
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], z, color="#475569", lw=0.8)
        for xx, yy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            ax.plot([xx, xx], [yy, yy], [0, 6], color="#475569", lw=0.8)
        if len(P_couv):
            ax.scatter(P_couv[:, 0], P_couv[:, 1], P_couv[:, 2], s=2.5, c="#7dd3fc", alpha=0.22, linewidths=0)
        if len(P_occ):
            ax.scatter(P_occ[:, 0], P_occ[:, 1], P_occ[:, 2], s=7, c="#cbd5e1", alpha=0.95, linewidths=0, marker="s")
        for r in d["racks"]:
            xs, ys = [r["x"][0], r["x"][1], r["x"][1], r["x"][0], r["x"][0]], [r["y"][0], r["y"][0], r["y"][1], r["y"][1], r["y"][0]]
            for z in (0.0, 6.0):
                ax.plot(xs, ys, z, color="#ef4444", lw=0.7, alpha=0.6)
            for xx, yy in zip(xs[:4], ys[:4]):
                ax.plot([xx, xx], [yy, yy], [0, 6], color="#ef4444", lw=0.7, alpha=0.6)
        if len(pistes):
            ax.scatter(pistes[:, 0], pistes[:, 1], pistes[:, 2], s=9, c="#f59e0b", alpha=0.8, linewidths=0)
        if len(lus):
            ax.scatter(lus[:, 0], lus[:, 1], lus[:, 2], s=26, c="#22c55e", linewidths=0, depthshade=False)
        if len(traj):
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="#60a5fa", lw=0.9, alpha=0.9)
            ax.scatter(*traj[-1], s=60, c="#3b82f6")
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(0, 6)
        ax.set_box_aspect((x1 - x0, y1 - y0, 12))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(titre, color="#c9d4e3", fontsize=11, pad=0)
    R = d["resume"]
    fig.text(0.5, 0.04,
             f"gris : obstacles vus par le lidar   ·   bleu : vu d'assez près pour lire   ·   vert : {R['codes_lus']} codes lus"
             f"   ·   orange : {R['pistes']} pistes   ·   rouge : les racks vrais   ·   {R['part_connue'] * 100:.0f} % de l'entrepôt connu",
             ha="center", color="#c9d4e3", fontsize=10)
    fig.subplots_adjust(left=0, right=1, top=0.98, bottom=0.06, wspace=0)
    fig.savefig(sortie, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return sortie


def genere(carte: mapping.Carte, vol: dict, sortie: Path) -> Path:
    html = GABARIT.replace("__DONNEES__", json.dumps(donnees(carte, vol), separators=(",", ":")))
    sortie.write_text(html)
    return sortie


if __name__ == "__main__":
    carte = mapping.Carte.charge(HERE / "carte")
    vol = json.loads((HERE / "vol.json").read_text())
    print(genere(carte, vol, HERE / "carte_3d.html"))
    print(image(carte, vol, HERE / "carte_3d.png"))
