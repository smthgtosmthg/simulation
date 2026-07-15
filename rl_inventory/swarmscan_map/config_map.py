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
    """

    read_distance_m: float = 1.25
    view_angle_deg: float = 50.0
    camera_fov_deg: float = 30.0
    max_speed_mps: float = 0.6
    max_yawrate_rps: float = 0.8
    dwell_steps: int = 2

    adr_read_distance_m: tuple[float, float] = (4.0, 1.25)
    adr_view_angle_deg: tuple[float, float] = (60.0, 50.0)
    adr_max_speed_mps: tuple[float, float] = (1.5, 0.6)   # contraignant dès L0 : pas de falaise au niveau 5 (cap 2.0 croisait la croisière 1.5 à L5)
    adr_max_yawrate_rps: tuple[float, float] = (1.5, 0.8)
    adr_dwell_steps: tuple[int, int] = (1, 2)


@dataclass
class CurriculumConfig:
    """ADR bidirectionnel piloté par le taux de lecture, + spawns dirigés + panne de drone."""

    notches: int = 12                     # crans FINS entre borne tolérante et nominale (6 = falaises → yo-yo observé)
    success_hi: float = 0.85              # BARRE DE MAÎTRISE, identique à tous les niveaux (décision utilisatrice : pas de promotion sans vrai bon pourcentage — 0.60 accepté avant = erreur)
    hi_decay_per_notch: float = 0.0
    hi_min: float = 0.85
    lo_gap: float = 0.25                  # plancher de recul = 0.60
    lo_min: float = 0.60
    promo_margin: float = 0.03            # promotion seulement si EMA > 0.88 TENUE 50 épisodes (anti-bruit)
    confirm_episodes: int = 50
    ema_alpha: float = 0.05
    min_episodes_per_notch: int = 300     # patience minimale avant de monter
    min_episodes_down: int = 800          # délai de GRÂCE : à l'arrivée d'un niveau dur, la moyenne chute mécaniquement — on laisse ~800 épisodes d'adaptation avant tout recul (sinon ping-pong)
    spawn_near_prob: tuple[float, float] = (0.7, 0.3)   # plancher 0.3 : garder une source d'événements de lecture à tous les niveaux
    dropout_prob: tuple[float, float] = (0.0, 0.30)     # panne : ramp APRÈS gate nominal
    dropout_after_nominal: bool = True
    episode_s_per_notch: float = 5.0      # budget d'épisode : 120 s + 5 s/cran (plafond ci-dessous) — gate serré = mission physiquement plus longue
    episode_s_max: float = 180.0


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
    scan_range_m: float = 2.5             # portée de l'empreinte de couverture — FIXE (découplée du gate de lecture :
    scan_speed_mps: float = 2.0           # couplée, l'économie fondait en r² avec le niveau et dépasser le cap de
    scan_yawrate_rps: float = 2.0         # vitesse supprimait la taxe d'overlap = ne pas lire devenait rentable)

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
    overlap_penalty: float = 0.05         # re-scan de cellules déjà couvertes par l'équipe (<90 % de couverture)
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
    action_diff: float = 0.05             # ‖a_t − a_{t−1}‖² sur actions écrêtées à ±3 (borné : sinon spirale numérique)
    bound_penalty: float = 0.5            # rappel vers [−1,1] : (|a|−1)₊² — restaure le gradient que le clamp silencieux supprime
    separation_m: float = 0.6             # pénalité si coéquipier plus proche
    separation_penalty: float = 1.0
    scan_norm: float = 52.0               # normalisation des comptes de cellules (empreinte fixe 2,5 m × 60° ≈ 52 cellules)


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

    episode_length_s: float = 120.0       # gate 1,25 m ⇒ couverture ~2× plus lente qu'à 3 m
    mission_target: float = 0.95          # fraction des tags LISIBLES (actifs & atteignables)
    gamma: float = 0.995
    lam: float = 0.95
    lidar_sectors: int = 72               # vecteur réactif : min par secteur de 5° (capteur 1800 intact)
    init_noise_std: float = 0.5
    entropy_coef: float = 0.01            # 0.003 → std mort en <1000 iters ; la visée est un geste de précision, il faut explorer longtemps


@dataclass
class SwarmScanMapConfig:
    gate: GateConfig = field(default_factory=GateConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    map: MapConfig = field(default_factory=MapConfig)
    reward: RewardMapConfig = field(default_factory=RewardMapConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    train: TrainMapConfig = field(default_factory=TrainMapConfig)


MAP_CFG = SwarmScanMapConfig()
