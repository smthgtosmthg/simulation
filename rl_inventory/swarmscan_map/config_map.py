"""Paramètres SwarmScan-Map (docs/conception_solution_finale.md)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateConfig:
    """Gate de lecture : seuils nominaux MESURÉS (calibrate_gate.py, 2026-07-09) + plages ADR.

    Calibration sur cv2.QRCodeDetectorAruco + upscale ×3 (décodeur famille Pore), caméra
    déclarée 1280×960, HFOV 60° : décodage jusqu'à 1,5 m (seuil 1,25 m avec marge) et 55°
    d'incidence (seuil 50°). Vitesse : loi de Cristiani 2020, seuil déclaré 0,6 m/s.
    camera_fov_deg = DEMI-angle du cône (HFOV 60° → 30).

    DEUX caméras LATÉRALES (axes ±90° du cap, mêmes intrinsèques calibrées chacune) :
    la lecture se fait en longeant les racks. Pivot décidé le 2026-07-24 — la visée fine
    d'une caméra frontale vivait dans le bruit d'exploration (0,91 bruité → 0,11 déterministe)
    et la faisabilité latérale est prouvée géométriquement sur les vraies configs (1,00).
    """

    read_distance_m: float = 1.25
    view_angle_deg: float = 50.0
    camera_fov_deg: float = 30.0
    max_speed_mps: float = 0.6
    max_yawrate_rps: float = 0.8
    dwell_steps: int = 2

    adr_read_distance_m: tuple[float, float] = (4.0, 1.25)
    adr_view_angle_deg: tuple[float, float] = (60.0, 50.0)
    # Le cran 0 DOIT être vacant sur l'axe vitesse : c'est le rôle d'un ADR, isoler une
    # difficulté à la fois. Avec (0.75, 0.60) et speed_uses_vz, seules 22 % des actions
    # satisfaisaient déjà le gate au cran 0 (11 % au nominal) : la géométrie ET la vitesse
    # devaient être apprises ensemble dès le premier épisode, et le curriculum n'a jamais
    # promu (34 runs). Borne haute = au-dessus du maximum physique 3D sqrt(1²+1²)=1.414.
    adr_max_speed_mps: tuple[float, float] = (1.45, 0.60)
    adr_max_yawrate_rps: tuple[float, float] = (1.6, 0.8)
    # le dwell est une PROPRIÉTÉ DU CAPTEUR (calibration : 2 images), jamais un paramètre
    # de curriculum. Un dwell de 1 pas = 33 ms est physiquement indéfendable.
    adr_dwell_steps: tuple[int, int] = (2, 2)
    # le flou de bougé est la vitesse dans le PLAN IMAGE : avec des caméras LATÉRALES un
    # mouvement vertical floute autant qu'un mouvement horizontal. Sans ça, l'action
    # (0,0,1,0) donnait une vitesse « nulle » au gate en montant réellement à 1,5 m/s.
    speed_uses_vz: bool = True
    # aucun crédit de lecture pendant les N premiers pas : au reset v = 0, donc le gate de
    # vitesse est satisfait d'emblée et les spawns dirigés offraient des lectures gratuites.
    credit_blackout_steps: int = 15


@dataclass
class CurriculumConfig:
    """ADR bidirectionnel piloté par le taux de lecture, + spawns dirigés + panne de drone."""

    notches: int = 10                     # l'axe vitesse balaie maintenant 1.45 → 0.60 : des crans
                                          # plus fins évitent les marches infranchissables
    # BARRE DE PROMOTION, à ne pas confondre avec la barre de qualité du mémoire.
    # Elle dit « ce cran est acquis, passe au suivant », pas « la mission est réussie ».
    # À 0.85 + marge 0.03 il fallait une EMA > 0.88 tenue 25 épisodes : simulation Monte-Carlo
    # sur la distribution mesurée (moyenne 0.39) → probabilité 0.00 %, et encore 0.00 % même
    # à une moyenne de 0.75. Le curriculum était mathématiquement inatteignable.
    # La barre de 0.85 reste le critère d'ACCEPTATION FINALE, mesurée au gate nominal par eval_map.py.
    # BARRE HAUTE ET PLATE (décision utilisatrice du 2026-07-27). Motif mesuré : promu à 0,62
    # le modèle ne tenait que 0,551 rétro-mesuré au cran 0, et sa courbe MONTAIT ENCORE au moment
    # de la promotion (0,28 → 0,60 sans plateau) : on ne l'a jamais laissé finir d'apprendre.
    # Exiger 0,88 partout force la maîtrise avant de changer les conditions.
    success_hi: float = 0.85
    hi_decay_per_notch: float = 0.0
    hi_min: float = 0.85
    lo_gap: float = 0.20                  # plancher de recul = 0.40
    lo_min: float = 0.35
    promo_margin: float = 0.03            # promotion si EMA > 0.88 TENUE 25 épisodes (anti-bruit)
    confirm_episodes: int = 25
    ema_alpha: float = 0.05
    min_episodes_per_notch: int = 120     # patience minimale avant de monter
    min_episodes_down: int = 250          # délai de GRÂCE avant tout recul (sinon ping-pong)
    spawn_near_prob: tuple[float, float] = (0.7, 0.3)   # plancher 0.3 : garder une source d'événements de lecture à tous les niveaux
    dropout_prob: tuple[float, float] = (0.0, 0.30)     # panne : ramp APRÈS gate nominal
    dropout_after_nominal: bool = True
    episode_s_per_notch: float = 15.0     # 10 crans × 15 s : 150 s au cran 0 → 295 s (plafond) au nominal
    episode_s_max: float = 295.0          # plan optimal au nominal : 207 s avec inertie


@dataclass
class MapConfig:
    """Cartes égocentriques multi-échelles construites en ligne (mémoire propre de l'essaim)."""

    bounds_x_m: tuple[float, float] = (-18.0, 18.0)
    bounds_y_m: tuple[float, float] = (-13.0, 13.0)
    cell_m: float = 0.25                  # grille globale
    crop_px: int = 32
    crop_spans_m: tuple[float, ...] = (8.0, 32.0)   # échelles égocentriques
    height_bands_m: tuple[float, ...] = (0.0, 1.5, 3.0, 4.7)  # bandes verticales des façades
    explored_radius_m: float = 3.0
    traj_decay: float = 0.98
    scan_range_m: float = 2.5             # portée de l'empreinte de couverture — FIXE (découplée du gate de lecture)
    # ces deux seuils doivent rester NON CONTRAIGNANTS (c'était l'intention écrite ici).
    # Ils ne l'étaient plus depuis max_lin_vel_mps 1.5 → 1.0 : la vitesse 3D maximale est
    # sqrt(1²+1²) = 1.414 m/s, donc « ≤ 1.0 » coupait la carte de couverture 47,7 % du temps.
    # La mémoire spatiale du drone était vide une action sur deux.
    scan_speed_mps: float = 1.5
    scan_yawrate_rps: float = 1.6

    @property
    def n_bands(self) -> int:
        return len(self.height_bands_m) - 1

    @property
    def grid_wh(self) -> tuple[int, int]:
        w = int(round((self.bounds_x_m[1] - self.bounds_x_m[0]) / self.cell_m))
        h = int(round((self.bounds_y_m[1] - self.bounds_y_m[0]) / self.cell_m))
        return w, h

    @property
    def n_channels(self) -> int:
        # occ, explored, scan×bandes, frontière, qr_lus, traj_soi, traj_coéquipiers
        return 6 + self.n_bands

    @property
    def obs_dim(self) -> int:
        return len(self.crop_spans_m) * self.n_channels * self.crop_px * self.crop_px


@dataclass
class RewardMapConfig:
    """Récompense v2 (conception §5) — dense en couverture, potentiel privilégié, jamais de temps constant."""

    facade_gain: float = 1.0              # cellules neuves ADJACENTES à un obstacle observé (façades)
    area_gain: float = 0.1                # cellules neuves quelconques
    marginal_gain: float = 0.5            # contribution marginale individuelle (anti-agglutinement)
    overlap_penalty: float = 0.02         # re-scan de cellules déjà couvertes par l'équipe (<90 % de couverture)
                                          # abaissé avec scan_norm : la taxe reste ~0,05 pt/pas/drone
    overlap_off_coverage: float = 0.90
    new_qr: float = 25.0                  # la lecture = revenu DOMINANT (~50 tags × 25 ≫ couverture) : la visée devient la compétence rentable
    new_qr_nominal: float = 10.0          # posture nominale tenue sur un tag (1×/tag/épisode) — le détour doit valoir une lecture
    read_gain_rise: float = 2.0           # lectures × (1 + rise·t) : une lecture au nominal coûte ~4 s de manœuvre, elle paye 30
    coverage_fade: float = 0.5            # couverture × (1 − fade·t) : la pression bascule balayer → lire
    milestones: tuple[float, ...] = (0.5, 0.75, 0.9)
    milestone_bonus: float = 10.0
    shaping_scale: float = 0.5            # potentiel privilégié γΦ'−Φ (ENTRAÎNEMENT seulement)
    shaping_clip_m: float = 12.0
    slow_potential: float = 0.3           # composante du potentiel : être LENT près d'un tag non lu (gradient dense vers la posture nominale)
    collision_scale: float = 2.0          # barrière graduée sous safe_distance, jamais terminale
    safe_distance_m: float = 0.5
    contact_penalty: float = 5.0          # en-dessous de collision_distance
    collision_distance_m: float = 0.25
    # ‖a_t − a_{t−1}‖² est calculé sur l'action ÉCHANTILLONNÉE : E‖Δa‖² = 2·4·σ² + ‖Δµ‖².
    # À σ = 0,29 le terme de bruit vaut à lui seul 0,05·8·0,29²·4500·3 = −454/épisode, contre
    # −486 mesuré : cette pénalité taxait le BRUIT D'EXPLORATION, pas la brusquerie du vol.
    # Son gradient sur σ vaut −3132 par unité de σ, d'où l'effondrement d'entropie 2,90 → 0,41
    # en 1600 itérations. Divisée par 10 : la taxe tombe à −45/épisode et cesse d'écraser σ.
    action_diff: float = 0.005
    bound_penalty: float = 0.5            # rappel vers [−1,1] : (|a|−1)₊² — restaure le gradient que le clamp silencieux supprime
    separation_m: float = 0.6             # barrière GRADUÉE (comme les obstacles), pas une marche :
    separation_penalty: float = 0.25      # à 1.0 en marche, elle coûtait −1098/épisode dans les BONS
                                          # épisodes contre −111 dans les mauvais (mesuré) et annulait
                                          # à elle seule le revenu des lectures (+1142)
    # UNITÉ D'AIRE FIXE, pas la taille de l'empreinte courante. Normaliser par l'empreinte
    # instantanée (105 cellules pour deux cônes latéraux de 2,5 m) rend la récompense
    # INVERSEMENT proportionnelle à la portée du capteur : le passage frontal 1,25 m → latéral
    # 2,5 m a fait passer scan_norm de ~13 à 105 et donc DIVISÉ PAR 8 tout le revenu de
    # couverture, sans que new_qr soit rééquilibré. Résultat mesuré : couvrir l'arène entière
    # rapportait 124 points contre 111 tags × 25 = 2775 — la récompense dense qui justifiait
    # toute la conception v2 pesait moins que 5 lectures. Avec 13.0, couvrir une bande complète
    # vaut ~1000 points : c'est de nouveau un gradient exploitable dès le premier épisode.
    scan_norm: float = 13.0


@dataclass
class LayoutConfig:
    """Générateur procédural de configurations d'inventaire + split train/val/test GELÉ."""

    active_frac: tuple[float, float] = (0.95, 1.0)  # ~chaque carton porte un tag (réaliste) — des façades-objectifs indiscernables des autres plafonnaient les lectures à ~0.4
    preread_frac: tuple[float, float] = (0.0, 0.0)  # ZÉRO tag pré-lu fantôme : « façade non scannée » = « travail restant », l'observation redevient complète
    train_configs: int = 500
    val_configs: int = 20
    test_configs: int = 20
    base_seed: int = 1234
    max_tag_z_m: float = 4.55             # alt_max 4.0 + 1.25·sin(30°) : au-dessus = inatteignable (exclu du dénominateur)


@dataclass
class TrainMapConfig:
    """Boucle d'entraînement."""

    episode_length_s: float = 150.0       # plan optimal au cran 0 = 131 s avec dwell 2 et cap
                                          # 0,75 m/s : 120 s était physiquement infaisable
    mission_target: float = 0.95          # fraction des tags LISIBLES (actifs & atteignables)
    gamma: float = 0.995
    lam: float = 0.95
    lidar_sectors: int = 72               # vecteur réactif : min par secteur de 5° (capteur 1800 intact)
    init_noise_std: float = 0.5
    entropy_coef: float = 0.01            # PHASE MONTÉE (défaut). Phase CONVERGENCE : lancer avec --entropy 0.001 — le bruit permanent faisait la visée à la place de la moyenne (testé : déterministe 0.05 vs bruité 0.66) ; en fin de montée on le retire pour forcer le transfert.


@dataclass
class SwarmScanMapConfig:
    gate: GateConfig = field(default_factory=GateConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    map: MapConfig = field(default_factory=MapConfig)
    reward: RewardMapConfig = field(default_factory=RewardMapConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    train: TrainMapConfig = field(default_factory=TrainMapConfig)


MAP_CFG = SwarmScanMapConfig()
