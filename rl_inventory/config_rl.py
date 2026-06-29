

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LidarConfig:
    """Grille de profondeur azimut × élévation (doc §3.1).

    360 colonnes (1 mesure/degré, tour complet) × 5 bandes de hauteur.
    Implémentée via RayCasterCfg + LidarPatternCfg (warp GPU, vectorisé).
    """

    num_azimuth: int = 360          # colonnes (1°/colonne)
    num_channels: int = 5           # bandes de hauteur
    horizontal_fov_deg: tuple[float, float] = (0.0, 360.0)
    vertical_fov_deg: tuple[float, float] = (-15.0, 15.0)  # couvre les hauteurs d'étagères
    horizontal_res_deg: float = 1.0  # 360 / 1.0 = 360 colonnes
    max_distance_m: float = 8.0      # portée LiDAR (cf. AIF existant)
    offset_z_m: float = 0.0          # capteur centré sur le corps du drone
    ray_alignment: str = "yaw"       # les rayons suivent le cap (yaw) du drone, restent à plat
    # bruit capteur (réglable) — idée utilisateur ; OFF par défaut pour le 1er test
    noise_std_m: float = 0.0         # écart-type du bruit gaussien sur les distances (m)
    dropout_prob: float = 0.0        # proba qu'un rayon ne renvoie rien (absorption / spéculaire)

    @property
    def num_rays(self) -> int:
        return self.num_azimuth * self.num_channels


@dataclass
class ActionConfig:
    """Commande de vitesse continue en repère drone (doc §3.2).

    a = (vx, vy, vz, yaw_rate), bornée. Niveau vitesse : le sim suit la
    consigne (write_root_velocity_to_sim). vz libre entre min/max altitude.
    """

    max_lin_vel_mps: float = 1.5     # borne |vx|,|vy|,|vz|
    max_yaw_rate_rps: float = 1.5    # borne yaw_rate
    altitude_min_m: float = 0.3      # le drone monte/descend pour scanner les étagères
    altitude_max_m: float = 3.0

    @property
    def dim(self) -> int:
        return 4


@dataclass
class RewardConfig:
    """Gabarit de récompense (doc §3.4) — valeurs à calibrer."""

    new_qr: float = 10.0             # un tag pas encore lu devient lu
    mission_complete: float = 100.0  # tous les QR lus
    collision: float = -5.0          # PAR PAS tant qu'en collision (lidar mini < seuil), pas de fin
    time_step: float = -0.1          # pénalité temps / pas (pousse à finir vite)
    jerk_lambda: float = 0.01        # -λ·‖Δaction‖² (fluidité, λ petit)
    approach_scale: float = 1.0      # +x par mètre gagné vers le QR non lu (plafonné à la distance de lecture)
    posture_scale: float = 0.5       # +x/pas quand le drone est BIEN PLACÉ pour lire (proche+face+LENT) un QR non lu
    proximity_penalty: float = 0.0   # (option) rester à distance de sécurité
    safe_distance_m: float = 0.5     # seuil proximité obstacle
    collision_distance_m: float = 0.25  # distance LiDAR mini en-dessous = collision


@dataclass
class QRProxyConfig:
    """Modèle « QR lu » : proxy géométrique pour l'entraînement (doc §4).

    Un tag est compté lu si : dans le FOV caméra + assez proche + bon angle +
    ligne de vue dégagée. (Le vrai décodeur pyzbar n'est utilisé qu'à l'éval.)
    """

    n_tags: int = 12                 # nombre de tags QR (taille mission, doc §11)
    max_read_distance_m: float = 3.0  # distance max de lecture
    max_view_angle_deg: float = 35.0  # angle max par rapport à la normale du tag
    camera_fov_deg: float = 60.0      # demi-angle du cône caméra
    min_dwell_steps: int = 2          # durée de visée minimale (pas consécutifs)
    max_read_speed_mps: float = 0.6   # vitesse max pour lire (sinon flou) — à calibrer sur pyzbar
    max_read_yawrate_rps: float = 0.8  # taux de lacet max pour lire
    check_line_of_sight: bool = True  # raycast drone→tag contre la géométrie (raffinement ultérieur)
    coverage_target: float = 1.0      # cible de couverture (1.0 = tous les QR)


@dataclass
class SceneConfig:
    """Entrepôt & layout (doc §5).

    Espace intérieur rectangulaire, rangées d'étagères formant des allées,
    tags QR sur les faces des racks à plusieurs hauteurs.

    Décision (contrainte RayCaster = 1 mesh statique) : racks/murs FIXES (vus
    par le LiDAR) + inventaire (boxes/tags) randomisé par épisode (le « QR lu »
    s'appuie sur la pose des boxes, pas le LiDAR).
    """


    warehouse_usd: str = (
        "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
    )
    env_size_xy_m: tuple[float, float] = (30.0, 20.0)
    env_spacing_m: float = 40.0      # > empreinte entrepôt (envs isolés)
    num_racks: int = 4
    randomize_inventory: bool = True  # positions boxes/tags variables / épisode
    randomize_racks: bool = False     # True => rebuild warp-mesh / reset (coûteux)


@dataclass
class TrainConfig:
    """Boucle de simulation / entraînement (doc §9)."""

    num_envs: int = 16               # à benchmarker sur 8 Go (entrepôt ≠ cartpole)
    decimation: int = 4              # pas physiques / pas de contrôle
    physics_dt: float = 1.0 / 120.0  # 120 Hz physique → contrôle ~30 Hz
    episode_length_s: float = 45.0   # budget pour scanner l'entrepôt
    num_drones: int = 3              # essaim : 3 drones par entrepôt
    seed: int = 42

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.decimation


@dataclass
class RLConfig:
    """Config racine de l'arène RL."""

    lidar: LidarConfig = field(default_factory=LidarConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    qr: QRProxyConfig = field(default_factory=QRProxyConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Instance par défaut, importée partout.
CFG = RLConfig()
