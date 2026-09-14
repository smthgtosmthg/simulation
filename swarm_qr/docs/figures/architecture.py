"""La figure d'architecture du chapitre 3 : composants, échanges et boucle de décision.

    architecture.py               avec les technologies nommées
    architecture.py --sans-techs  sans elles, si le chapitre de conception doit rester abstrait
"""
import sys
from pathlib import Path

TECHS = "--sans-techs" not in sys.argv

W, H = 1560, 1132
C = {"fond": "#F6F8F7", "cadre": "#C9D3CF", "encre": "#16201C", "douce": "#5F6B66",
     "percep": "#0B7285", "carte": "#B07A22", "decision": "#364FC7", "guide": "#7A3E9D",
     "super": "#6B7670", "lu": "#2F9E44", "alerte": "#C92A2A", "signal": "#E8590C"}
out = []


def texte(x, y, s, taille=11.5, couleur=None, gras=False, ancre="start", italique=False):
    st = f'font-size="{taille}" fill="{couleur or C["encre"]}" text-anchor="{ancre}"'
    if gras:
        st += ' font-weight="650"'
    if italique:
        st += ' font-style="italic"'
    out.append(f'<text x="{x}" y="{y}" {st}>{s}</text>')


def cadre(x, y, w, h, titre, lignes, couleur, ombre=False, note=None):
    if ombre:
        for d in (10, 5):
            out.append(f'<rect x="{x+d}" y="{y+d}" width="{w}" height="{h}" rx="10" '
                       f'fill="#FFFFFF" stroke="{C["cadre"]}" stroke-width="1.2"/>')
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#FFFFFF" '
               f'stroke="{couleur}" stroke-width="1.8"/>')
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="30" rx="10" fill="{couleur}" opacity="0.12"/>')
    out.append(f'<rect x="{x}" y="{y+20}" width="{w}" height="10" fill="{couleur}" opacity="0.12"/>')
    texte(x + 14, y + 20, titre, 12.5, couleur, gras=True)
    if note:
        texte(x + w - 14, y + 20, note, 10.5, C["douce"], ancre="end", italique=True)


def sous(x, y, w, h, titre, lignes, couleur):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{couleur}" opacity="0.055"/>')
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="none" '
               f'stroke="{couleur}" stroke-width="1" stroke-opacity="0.45"/>')
    texte(x + 12, y + 19, titre, 11.5, couleur, gras=True)
    yy = y + 38
    for l in lignes:
        if l.startswith("~"):
            if not TECHS:
                continue
            yy += 4
            texte(x + 25, yy, l[1:], 9.8, C["douce"], italique=True)
        else:
            if not l.startswith(" "):
                out.append(f'<circle cx="{x+16}" cy="{yy-4}" r="1.9" fill="{couleur}" opacity="0.7"/>')
            texte(x + 25, yy, l, 10.8)
        yy += 16


def fleche(points, couleur, pointille=False, epais=2.0):
    d = "M " + " L ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="6 4"' if pointille else ""
    out.append(f'<path d="{d}" fill="none" stroke="{couleur}" stroke-width="{epais}"{dash} '
               f'marker-end="url(#t{couleur[1:]})" stroke-linejoin="round"/>')


def eti(x, y, lignes, couleur):
    """Une étiquette posée sur un trait : fond blanc, une ou plusieurs lignes centrées."""
    lignes = [lignes] if isinstance(lignes, str) else lignes
    lg = max(len(s) for s in lignes) * 5.7 + 18
    ht = 15 * len(lignes) + 6
    out.append(f'<rect x="{x-lg/2}" y="{y-ht/2}" width="{lg}" height="{ht}" rx="4" '
               f'fill="#FFFFFF" stroke="{couleur}" stroke-width="0.9" stroke-opacity="0.55"/>')
    for i, s in enumerate(lignes):
        texte(x, y - ht / 2 + 14 + 15 * i, s, 10.3, couleur, ancre="middle", gras=True)


# ------------------------------------------------------------------ colonnes
AX, AW = 40, 376        # véhicule
BX, BW = 540, 420       # carte partagée
CX, CW = 1060, 460      # décision
CA, CB = 478, 1010      # les deux couloirs verticaux libres

texte(W / 2, 42, "Architecture of the proposed system", 19, C["encre"], gras=True, ancre="middle")
texte(W / 2, 64, "one shared map, N vehicles, and a decision loop that never commits before it observes",
      12, C["douce"], ancre="middle", italique=True)

# --- l'entrepôt : ce que le système subit et ne choisit pas ----------------
cadre(AX, 86, CX + CW - AX, 66, "The warehouse, unknown in advance", [], C["encre"],
      note="Isaac Sim 5.1 + Pegasus + ArduPilot SITL, physics at 1/800 s" if TECHS
      else "simulated, with the autopilots in the loop")
texte(AX + 16, 132,
      "racks and cartons placed differently in every building   ·   labels on the faces of the cartons, on both sides of an aisle   ·   "
      "an obstacle may appear while the mission runs   ·   a vehicle may be lost", 10.8, C["encre"])

# --- A : le véhicule -------------------------------------------------------
cadre(AX, 176, AW, 560, "Aerial vehicle", [], C["percep"], ombre=True, note="one of N")
sous(AX + 16, 216, AW - 32, 126, "Sensors", [
    "two lateral cameras, high resolution, for reading",
    "one forward camera",
    "3D LiDAR, full revolution",
    "vehicle pose, from the autopilot",
    "~Isaac Sim native : 2 x 1024x768 at 60 deg, LiDAR 360x36 deg"], C["percep"])
sous(AX + 16, 356, AW - 32, 160, "Perception", [
    "learned spotter : finds cartons and labels",
    "     at range, without decoding them",
    "decoder : reads a label from close range only",
    "range along the ray : places a label in 3D",
    "LiDAR rays : free volume, solid volume",
    "~YOLO11n at 1024 px   ·   zxing-cpp   ·   range + solvePnP"], C["percep"])
sous(AX + 16, 530, AW - 32, 142, "Flight controller", [
    "transit along the given path",
    "slow approach, then hold in front of the face",
    "abandons on a cut path or on lack of progress",
    "velocity commands down to the autopilot",
    "~MAVLink velocity setpoints, ArduPilot SITL in GUIDED"], C["percep"])
texte(AX + 16, 698, "runs at flight rate, on board", 10.5, C["douce"], italique=True)
texte(AX + 16, 716, "the N vehicles run this block independently", 10.5, C["douce"], italique=True)

# --- B : la carte partagée -------------------------------------------------
cadre(BX, 326, BW, 410, "Shared map", [], C["carte"], note="a single instance, common to the team")
sous(BX + 16, 366, BW - 32, 78, "Occupancy", [
    "solid / free / not yet observed",
    "~numpy grid at 25 cm   ·   A* over the cost grid"], C["carte"])
sous(BX + 16, 456, BW - 32, 78, "Oriented coverage", [
    "observed, and from which direction"], C["carte"])
sous(BX + 16, 546, BW - 32, 96, "Label table", [
    "unknown  →  spotted, unread  →  read",
    "position, orientation, identifier"], C["carte"])
sous(BX + 16, 654, BW - 32, 66, "Teammates and claims", [
    "positions, and claims that expire"], C["carte"])

# --- C : la décision -------------------------------------------------------
cadre(CX, 326, CW, 410, "Decision", [], C["decision"], note="per vehicle, once per cycle")
sous(CX + 16, 366, CW - 32, 96, "Candidate generation", [
    "read a spotted label | cover an unseen face | explore",
    "all written the same way : a pose and a direction"], C["decision"])
sous(CX + 16, 474, CW - 32, 112, "Scoring on a single scale", [
    "+ expected return          − path cost",
    "− already claimed         − teammate nearby",
    "+ semantic advice, multiplied by its weight",
    "~our own code : a weight of zero leaves pure geometry"], C["decision"])
sous(CX + 16, 598, CW - 32, 78, "Feasibility and selection", [
    "a path through space known to be free, best first"], C["decision"])
texte(CX + 16, 702, "one decision per cycle, so the physics is never held up", 10.5, C["douce"], italique=True)

# --- guide -----------------------------------------------------------------
cadre(CX, 176, CW, 118, "Semantic guide", [], C["guide"], note="optional")
sous(CX + 16, 206, CW - 32, 84 if TECHS else 68, "", [
    "a pre-trained vision-language model, asked : which area next",
    "asynchronous ; a weight of zero removes it entirely",
    "~SmolVLM-500M / SmolVLM-2B / Qwen2-VL, 4-bit, LoRA optional"], C["guide"])

# --- supervision -----------------------------------------------------------
LARG = CX + CW - AX
cadre(AX, 828, LARG, 116, "Mission supervision", [], C["super"], note="outside the vehicles")
for k, (t, l) in enumerate([
        ("Rhythms", "flight control, decision and advice run at different rates"),
        ("Progress", "distinct identifiers read, against the expected inventory"),
        ("Termination", "share of the inventory reached, or no new identifier for a delay")]):
    sous(AX + 16 + k * 488, 866, 470, 62, t, [l], C["super"])

# ------------------------------------------------------------------ échanges
# 0. l'entrepôt -> les capteurs
fleche([(AX + AW * 0.45, 152), (AX + AW * 0.45, 172)], C["percep"], epais=1.6)

# 1. perception -> carte
fleche([(AX + AW, 430), (BX - 8, 430)], C["lu"])
eti(CA, 410, "observations", C["lu"])
texte(CA, 454, "codes, spotted labels,", 9.6, C["douce"], ancre="middle")
texte(CA, 467, "free and solid volume", 9.6, C["douce"], ancre="middle")

# 2. carte -> génération de candidats
fleche([(BX + BW, 400), (CX - 8, 400)], C["carte"])
eti(CB, 380, "what is known", C["carte"])

# 3. carte -> notation : revendications et coéquipiers
fleche([(BX + BW, 688), (CB, 688), (CB, 528), (CX - 8, 528)], C["carte"])
eti(CB, 608, ["claims,", "teammates"], C["carte"])

# 4. guide -> décision
fleche([(CX + CW * 0.70, 292), (CX + CW * 0.70, 322)], C["guide"])
eti(CX + CW * 0.70, 306, "advice, weighted", C["guide"])

# 5. carte -> guide
fleche([(BX + BW - 60, 326), (BX + BW - 60, 302), (CX + CW * 0.28, 302),
        (CX + CW * 0.28, 292)], C["guide"], pointille=True, epais=1.6)
eti(1000, 302, "summary of the map", C["guide"])

# 6. contrôleur -> carte : libération
fleche([(AX + AW, 600), (CA, 600), (CA, 692), (BX - 8, 692)], C["alerte"], epais=1.6)
eti(CA, 644, ["released", "on abandon"], C["alerte"])

# 7. sélection -> carte : la revendication
fleche([(CX + 60, 736), (CX + 60, 772), (BX + BW - 80, 772), (BX + BW - 80, 738)], C["signal"])
eti(CB, 772, "claim, expires unless renewed", C["signal"])

# 8. sélection -> contrôleur de vol
fleche([(CX + CW - 80, 736), (CX + CW - 80, 796), (AX + AW - 80, 796), (AX + AW - 80, 738)], C["signal"])
eti(700, 796, "chosen pose and path", C["signal"])

# 9. carte -> supervision
fleche([(BX + 40, 736), (BX + 40, 828)], C["super"], epais=1.5, pointille=True)
eti(BX + 40, 756, "progress", C["super"])

# ------------------------------------------------------------------ légende
LY = 974
out.append(f'<rect x="{AX}" y="{LY}" width="{LARG}" height="106" rx="10" '
           f'fill="#FFFFFF" stroke="{C["cadre"]}" stroke-width="1.2"/>')
texte(AX + 16, LY + 24, "The loop", 12.5, C["encre"], gras=True)
etapes = ["the vehicles observe,", "the map holds what is", "each vehicle prices the",
          "the controller flies it,", "what it establishes"]
etapes2 = ["and write into the map", "now worth doing", "actions and claims one", "or gives it up",
           "goes back into the map"]
x = AX + 20
for i, (e1, e2) in enumerate(zip(etapes, etapes2)):
    out.append(f'<circle cx="{x+11}" cy="{LY+50}" r="11" fill="{C["signal"]}" opacity="0.14"/>')
    texte(x + 11, LY + 54, str(i + 1), 11, C["signal"], gras=True, ancre="middle")
    texte(x + 28, LY + 47, e1, 10.8)
    texte(x + 28, LY + 61, e2, 10.8)
    if i < 4:
        texte(x + 272, LY + 54, "→", 14, C["douce"])
    x += 292
texte(AX + 20, LY + 88,
      "Nothing is written before it is observed, and nothing is claimed for longer than it is renewed : the same two rules "
      "cover a warehouse never seen before, an obstacle that appears mid-mission, and the loss of a vehicle.",
      10.8, C["douce"], italique=True)

# ------------------------------------------------------------------ écriture
marqueurs = "".join(
    f'<marker id="t{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" '
    f'markerHeight="6.5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{c}"/></marker>'
    for c in set(C.values()))
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
       f'font-family="Inter, Helvetica Neue, Arial, sans-serif">'
       f'<defs>{marqueurs}</defs><rect width="{W}" height="{H}" fill="{C["fond"]}"/>'
       + "".join(out) + "</svg>")
p = Path(__file__).with_name("architecture.svg")
p.write_text(svg)
print(f"{p.name} : {len(svg)/1024:.0f} ko, {W}x{H}")
