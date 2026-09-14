"""Les trois schémas de la section perception : la chaîne du composant, l'équivalence
distance-angle, et les deux régimes de placement.

    perception_schemas.py
"""
import math
from pathlib import Path

C = {"fond": "#F6F8F7", "cadre": "#C9D3CF", "encre": "#16201C", "douce": "#5F6B66",
     "percep": "#0B7285", "carte": "#B07A22", "lu": "#2F9E44", "appris": "#364FC7",
     "alerte": "#C92A2A", "gris": "#98A4A0"}


class Toile:
    def __init__(self, w, h):
        self.w, self.h, self.out = w, h, []

    def txt(self, x, y, s, t=11.5, col=None, gras=False, anc="start", ital=False):
        st = f'font-size="{t}" fill="{col or C["encre"]}" text-anchor="{anc}"'
        st += ' font-weight="650"' if gras else ""
        st += ' font-style="italic"' if ital else ""
        self.out.append(f'<text x="{x}" y="{y}" {st}>{s}</text>')

    def rect(self, x, y, w, h, col, r=8, fill="#FFFFFF", op=1.0, ep=1.6, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" '
                        f'fill-opacity="{op}" stroke="{col}" stroke-width="{ep}"{d}/>')

    def boite(self, x, y, w, h, titre, lignes, col):
        self.rect(x, y, w, h, col)
        self.out.append(f'<rect x="{x}" y="{y}" width="{w}" height="26" rx="8" fill="{col}" opacity="0.12"/>')
        self.out.append(f'<rect x="{x}" y="{y+18}" width="{w}" height="8" fill="{col}" opacity="0.12"/>')
        self.txt(x + 12, y + 18, titre, 11.5, col, gras=True)
        yy = y + 44
        for l in lignes:
            ital = l.startswith("~")
            self.txt(x + 13, yy, l[1:] if ital else l, 9.8 if ital else 10.5,
                     C["douce"] if ital else C["encre"], ital=ital)
            yy += 15

    def ligne(self, pts, col, ep=1.8, dash=None, fleche=True):
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        mk = f' marker-end="url(#f{col[1:]})"' if fleche else ""
        self.out.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{ep}"{da}{mk} '
                        f'stroke-linejoin="round" stroke-linecap="round"/>')

    def eti(self, x, y, lignes, col):
        lignes = [lignes] if isinstance(lignes, str) else lignes
        lg, ht = max(len(s) for s in lignes) * 5.5 + 16, 14 * len(lignes) + 8
        self.rect(x - lg / 2, y - ht / 2, lg, ht, col, r=4, ep=0.9)
        for i, s in enumerate(lignes):
            self.txt(x, y - ht / 2 + 13 + 14 * i, s, 10, col, anc="middle", gras=True)

    def ecrit(self, nom):
        mk = "".join(
            f'<marker id="f{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
            f'markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{c}"/></marker>' for c in set(C.values()))
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
               f'width="{self.w}" height="{self.h}" font-family="Inter, Helvetica, Arial, sans-serif">'
               f'<defs>{mk}</defs><rect width="{self.w}" height="{self.h}" fill="{C["fond"]}"/>'
               + "".join(self.out) + "</svg>")
        p = Path(__file__).with_name(nom)
        p.write_text(svg)
        print(f"{nom} : {len(svg)/1024:.0f} ko, {self.w}x{self.h}")


# ---------------------------------------------------------------- 1. la chaîne
def pipeline():
    t = Toile(1420, 560)
    t.txt(t.w / 2, 34, "Inside the perception component", 16, C["encre"], gras=True, anc="middle")
    t.txt(t.w / 2, 54, "one cycle, one vehicle", 11, C["douce"], anc="middle", ital=True)

    t.boite(40, 90, 250, 160, "Sensors", [
        "left camera", "right camera", "LiDAR revolution", "camera pose",
        "~paired with the previous instant :", "~the renderer is one frame behind"], C["percep"])
    t.boite(360, 100, 290, 96, "Decoding", [
        "identifier and four corners", "both cameras, every cycle"], C["percep"])
    t.boite(360, 250, 290, 96, "Detection", [
        "a region and a class", "one camera per cycle, alternating"], C["appris"])
    t.boite(730, 116, 300, 214, "Placement in 3D", [
        "pixel  \u2192  world ray", "range : the LiDAR, or the map past 8 m", "",
        "rejected if :", "~the incidence is too large",
        "~the range falls outside the envelope", "~the placement is implausible"], C["encre"])
    t.boite(1110, 90, 270, 300, "Shared map", [
        "READ", "~identifier, position, normal", "", "SPOTTED",
        "~position, direction of approach", "", "SEMANTIC", "~a carton is here", "",
        "OCCUPANCY", "~free and solid volume", "", "COVERAGE", "~observed, and from where"], C["carte"])

    t.ligne([(290, 148), (352, 148)], C["percep"])
    t.ligne([(290, 190), (320, 190), (320, 298), (352, 298)], C["appris"])
    t.ligne([(650, 148), (690, 148), (690, 176), (722, 176)], C["percep"])
    t.eti(692, 122, "corners", C["percep"])
    t.ligne([(650, 298), (690, 298), (690, 268), (722, 268)], C["appris"])
    t.eti(694, 322, "region", C["appris"])
    t.ligne([(290, 226), (330, 226), (330, 430), (880, 430), (880, 338)], C["gris"], ep=1.5, dash="5 4")
    t.eti(590, 430, "LiDAR point cloud", C["gris"])
    t.ligne([(1030, 172), (1070, 172), (1070, 148), (1102, 148)], C["lu"])
    t.eti(1068, 122, "read", C["lu"])
    t.ligne([(1030, 250), (1070, 250), (1070, 232), (1102, 232)], C["appris"])
    t.eti(1068, 278, "spotted", C["appris"])
    t.ligne([(290, 208), (306, 208), (306, 498), (1240, 498), (1240, 398)], C["carte"], ep=1.5, dash="5 4")
    t.eti(800, 498, "LiDAR and camera cone, written straight into the map", C["carte"])
    t.ecrit("fig_perception_chaine.svg")


# -------------------------------------------- 2. l'équivalence distance / angle
def apparente():
    """Un seul cône de vue, deux panneaux exactement inscrits dedans : l'un de face et loin,
    l'autre incliné et proche. Les positions sont calculées, pas dessinées à la main."""
    t = Toile(1040, 500)
    t.txt(t.w / 2, 34, "Why distance and angle are a single limit", 16, C["encre"], gras=True, anc="middle")
    t.txt(t.w / 2, 54, "a panel tilted by \u03b1 fills the same cone as an untilted panel 1 / cos \u03b1 times further away",
          11, C["douce"], anc="middle", ital=True)

    CX, CY = 110, 250
    TH = math.radians(8.0)          # demi-angle du cône
    AL = math.radians(45.0)         # inclinaison du panneau proche
    D = 330.0                       # distance du panneau incliné, sur l'axe
    DF = D / math.cos(AL)           # distance du panneau de face

    for sg in (-1, 1):
        t.ligne([(CX, CY), (CX + 630, CY + sg * 630 * math.tan(TH))], C["gris"], ep=1.1,
                dash="5 4", fleche=False)
    t.ligne([(CX + 14, CY), (CX + 620, CY)], C["douce"], ep=1.1, dash="4 4", fleche=False)

    # le panneau incliné : les deux bouts touchent exactement le cône
    ta = D * math.tan(TH) / (math.cos(AL) - math.sin(AL) * math.tan(TH))
    tb = D * math.tan(TH) / (math.cos(AL) + math.sin(AL) * math.tan(TH))
    ax, ay = CX + D + ta * math.sin(AL), CY - ta * math.cos(AL)
    bx, by = CX + D - tb * math.sin(AL), CY + tb * math.cos(AL)
    t.out.append(f'<line x1="{bx:.1f}" y1="{by:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" stroke="{C["appris"]}" '
                 f'stroke-width="9" stroke-linecap="round"/>')
    # le panneau de face, plus loin
    hf = DF * math.tan(TH)
    t.out.append(f'<line x1="{CX+DF:.1f}" y1="{CY-hf:.1f}" x2="{CX+DF:.1f}" y2="{CY+hf:.1f}" '
                 f'stroke="{C["lu"]}" stroke-width="9" stroke-linecap="round"/>')

    # ce que la caméra reçoit : la même ouverture
    xi = CX + 150
    hi = 150 * math.tan(TH)
    t.out.append(f'<line x1="{xi}" y1="{CY-hi:.1f}" x2="{xi}" y2="{CY+hi:.1f}" stroke="{C["encre"]}" '
                 f'stroke-width="7" stroke-linecap="round"/>')
    t.txt(xi, CY - hi - 14, "the image", 10.5, C["encre"], anc="middle", gras=True)
    t.txt(xi, CY + hi + 24, "one extent, two panels", 9.8, C["douce"], anc="middle", ital=True)

    t.out.append(f'<circle cx="{CX}" cy="{CY}" r="9" fill="{C["encre"]}"/>')
    t.txt(CX, CY + 30, "camera", 10.5, C["douce"], anc="middle")

    # les deux distances
    t.ligne([(CX, CY + 92), (CX + D, CY + 92)], C["appris"], ep=1.3)
    t.txt(CX + D / 2, CY + 86, "d", 13, C["appris"], anc="middle", gras=True, ital=True)
    t.ligne([(CX, CY + 128), (CX + DF, CY + 128)], C["lu"], ep=1.3)
    t.txt(CX + DF / 2, CY + 122, "d / cos \u03b1", 13, C["lu"], anc="middle", gras=True, ital=True)
    for x, col in ((CX + D, C["appris"]), (CX + DF, C["lu"])):
        t.out.append(f'<line x1="{x:.0f}" y1="{CY}" x2="{x:.0f}" y2="{CY+(92 if col==C["appris"] else 128)}" '
                     f'stroke="{col}" stroke-width="1" stroke-dasharray="3 3"/>')

    # l'angle
    t.out.append(f'<path d="M {CX+D:.0f} {CY-42} A 42 42 0 0 1 {CX+D+42*math.sin(AL):.0f} '
                 f'{CY-42*math.cos(AL):.0f}" fill="none" stroke="{C["appris"]}" stroke-width="1.5"/>')
    t.txt(CX + D + 16, CY - 50, "\u03b1", 14, C["appris"], gras=True)

    t.txt(CX + D, CY - 112, "tilted, closer", 11.5, C["appris"], gras=True, anc="middle")
    t.txt(CX + DF + 60, CY - 104, "face-on, further away", 11.5, C["lu"], gras=True, anc="middle")

    t.rect(320, 412, 400, 66, C["encre"], r=8)
    t.txt(520, 438, "the two impose the same limit, so one quantity carries both :",
          10.5, C["douce"], anc="middle")
    t.txt(520, 464, "d\u2090\u209a\u209a  =  d / cos \u03b1", 15, C["encre"], gras=True, anc="middle")
    t.ecrit("fig_distance_apparente.svg")


# -------------------------------------------------- 3. les deux régimes
def regimes():
    t = Toile(1420, 620)
    t.txt(t.w / 2, 34, "Placing a detection : two regimes, separated by the LiDAR itself", 16,
          C["encre"], gras=True, anc="middle")
    t.txt(t.w / 2, 54, "plan view of one aisle, not to scale", 11, C["douce"], anc="middle", ital=True)

    DX, DY, FACE = 110, 470, 200          # drone, et la face du rack
    ECH = 115                             # px par mètre
    t.rect(120, 96, 1270, FACE - 96, C["gris"], r=4, fill=C["gris"], op=0.16, ep=1.2)
    t.txt(140, 128, "rack", 12, C["douce"], gras=True)

    # les rayons du lidar : même pas angulaire, écart croissant sur la face
    for k in range(1, 14):
        th = math.radians(6 * k)
        x = DX + (DY - FACE) * math.tan(th)
        if x > 1380:
            break
        t.ligne([(DX, DY), (x, FACE)], C["gris"], ep=0.9, fleche=False)
        t.out.append(f'<circle cx="{x:.0f}" cy="{FACE}" r="2.6" fill="{C["gris"]}"/>')

    t.txt(760, 228, "the dots mark where the LiDAR rays land : their spacing grows with range",
          10.5, C["douce"], anc="middle", ital=True)
    t.out.append(f'<circle cx="{DX}" cy="{DY}" r="11" fill="{C["encre"]}"/>')
    t.txt(DX, DY + 34, "vehicle", 11, C["encre"], gras=True, anc="middle")

    def etiquette(x, col):
        t.rect(x - 16, FACE - 5, 32, 14, col, r=2, fill=col, op=0.9, ep=0)

    # près : un rayon tombe sur l'étiquette
    XP = DX + (DY - FACE) * math.tan(math.radians(30))
    etiquette(XP, C["lu"])
    t.ligne([(DX, DY), (XP, FACE + 4)], C["lu"], ep=2.6)
    t.eti(XP + 130, 330, ["a ray lands on the label", "range from the LiDAR"], C["lu"])

    # loin : aucun rayon ne tombe dessus
    XL = 1080
    etiquette(XL, C["appris"])
    t.ligne([(DX, DY), (XL, FACE + 4)], C["appris"], ep=2.2, dash="7 5")
    for k in range(6, 14):
        px = DX + (XL - DX) * k / 14.0
        py = DY + (FACE - DY) * k / 14.0
        t.rect(px - 7, py - 7, 14, 14, C["appris"], r=2, fill=C["appris"], op=0.14, ep=0.8)
    t.rect(XL - 9, FACE - 4, 18, 18, C["appris"], r=2, fill=C["appris"], op=0.35, ep=1.2)
    t.eti(700, 478, ["no ray lands on it : the nearest one", "reaches the neighbour, or a gap",
                         "range from the map, first occupied cell"], C["appris"])

    # la frontière
    XB = DX + math.sqrt((8 * ECH) ** 2 - (DY - FACE) ** 2)
    t.ligne([(XB, 96), (XB, 560)], C["alerte"], ep=1.6, dash="8 5", fleche=False)
    t.eti(XB, 578, "r = 8 m,  two rays are now further apart than one map cell", C["alerte"])
    t.txt(XB - 14, 88, "LiDAR trusted", 10.5, C["alerte"], anc="end", gras=True)
    t.txt(XB + 14, 88, "map takes over", 10.5, C["alerte"], gras=True)

    t.rect(200, 480, 330, 76, C["encre"], r=8)
    t.txt(216, 506, "ray separation grows with range", 11, C["douce"], ital=True)
    t.txt(216, 532, "δ(r)  =  r · Δθ", 13.5, C["encre"], gras=True)
    t.ecrit("fig_regimes.svg")


pipeline()
apparente()
regimes()
