#!/bin/bash
set -e

N_DRONES=${1:-3}
DRONE_SPACING=3      

WORKSPACE="$HOME/simulation_mc02"
ARDUPILOT_DIR="$HOME/ardupilot"
SIM_VEHICLE_SCRIPT="$ARDUPILOT_DIR/Tools/autotest/sim_vehicle.py"
SIM_VEHICLE_PYTHON="${SIM_VEHICLE_PYTHON:-/usr/bin/python3}"
PLUGIN_DIR="$HOME/ardupilot_gazebo"
IRIS_MODEL_SRC="$PLUGIN_DIR/models/iris_with_ardupilot"
MODELS_DIR="$WORKSPACE/models"
WORLDS_DIR="$WORKSPACE/worlds"
GENERATED_WORLD="$WORLDS_DIR/warehouse_drones.sdf"

echo "=============================================="
echo " [6] Multi-Drones : $N_DRONES drones"
echo "=============================================="

if [ ! -d "$ARDUPILOT_DIR" ]; then
    echo "ERREUR : ArudPilot non trouvé dans $ARDUPILOT_DIR"
    echo "Lance d'abord le script 03_install_px4_sitl.sh"
    exit 1
fi

if [ ! -f "$SIM_VEHICLE_SCRIPT" ]; then
  echo "ERREUR : sim_vehicle.py non trouvé dans $SIM_VEHICLE_SCRIPT"
  echo "Vérifie l'installation ArduPilot SITL (script 03_install_ardupilot_sitl.sh)."
  exit 1
fi

if ! command -v "$SIM_VEHICLE_PYTHON" >/dev/null 2>&1; then
  echo "ERREUR : interpréteur Python introuvable : $SIM_VEHICLE_PYTHON"
  echo "Définis SIM_VEHICLE_PYTHON ou vérifie ton installation système."
  exit 1
fi

if [ ! -d "$IRIS_MODEL_SRC" ]; then
    echo "ERREUR : Modèle iris non trouvé dans $IRIS_MODEL_SRC"
    echo "Lance d'abord le script 03_install_px4_sitl.sh"
    exit 1
fi

if [ ! -d "$WORLDS_DIR" ]; then
    echo "ERREUR : Dossier worlds/ non trouvé."
    echo "Lance d'abord le script 05_setup_warehouse.sh"
    exit 1
fi

mkdir -p "$MODELS_DIR" "$WORLDS_DIR"

echo ""
echo "Nettoyage des anciens processus..."
pkill -f arducopter 2>/dev/null || true
pkill -f "gz sim" 2>/dev/null || true
pkill -f mavproxy 2>/dev/null || true
pkill -f sim_vehicle 2>/dev/null || true
sleep 2

export GZ_SIM_SYSTEM_PLUGIN_PATH=$PLUGIN_DIR/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
export GZ_SIM_RESOURCE_PATH=$MODELS_DIR:$PLUGIN_DIR/models:$PLUGIN_DIR/worlds:$WORKSPACE/worlds:$GZ_SIM_RESOURCE_PATH

# Forcer le driver EGL NVIDIA pour éviter les warnings libEGL
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA

# Auto-détection du display X11 
if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    for sock in /tmp/.X11-unix/X*; do
        d=":${sock##*/tmp/.X11-unix/X}"
        if xdpyinfo -display "$d" >/dev/null 2>&1; then
            echo "  (DISPLAY=$DISPLAY invalide, basculement vers $d)"
            export DISPLAY="$d"
            break
        fi
    done
fi

# Créer les copies du modèle iris 
echo ""
echo "Création des modèles iris pour $N_DRONES drones..."

for i in $(seq 0 $((N_DRONES - 1))); do
    MODEL_NAME="iris_instance_$i"
    MODEL_DEST="$MODELS_DIR/$MODEL_NAME"
    PORT_IN=$((9002 + i * 10))
    PORT_OUT=$((9003 + i * 10))

    # Copier le modèle source
    rm -rf "$MODEL_DEST"
    cp -r "$IRIS_MODEL_SRC" "$MODEL_DEST"

    # Modifier les ports dans model.sdf
    sed -i "s|<fdm_port_in>[0-9]\+</fdm_port_in>|<fdm_port_in>$PORT_IN</fdm_port_in>|g" \
        "$MODEL_DEST/model.sdf"
    sed -i "s|<fdm_port_out>[0-9]\+</fdm_port_out>|<fdm_port_out>$PORT_OUT</fdm_port_out>|g" \
        "$MODEL_DEST/model.sdf"

    # Mettre à jour le nom dans model.config
    sed -i "s|<name>[^<]*</name>|<name>$MODEL_NAME</name>|" "$MODEL_DEST/model.config"

    # Injecter le capteur LiDAR (gpu_lidar) pour l'exploration
    python3 -c "
sdf = open('$MODEL_DEST/model.sdf').read()
lidar = '''
    <!-- LiDAR sensor for exploration -->
    <link name=\"lidar_link\">
      <pose>0 0 0.05 0 0 0</pose>
      <inertial>
        <mass>0.01</mass>
        <inertia>
          <ixx>0.000001</ixx><iyy>0.000001</iyy><izz>0.000001</izz>
        </inertia>
      </inertial>
      <visual name=\"lidar_visual\">
        <geometry><cylinder><radius>0.02</radius><length>0.04</length></cylinder></geometry>
        <material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse></material>
      </visual>
      <sensor name=\"lidar\" type=\"gpu_lidar\">
        <always_on>true</always_on>
        <update_rate>5</update_rate>
        <visualize>false</visualize>
        <lidar>
          <scan>
            <horizontal>
              <samples>360</samples>
              <resolution>1</resolution>
              <min_angle>-3.14159265</min_angle>
              <max_angle>3.14159265</max_angle>
            </horizontal>
            <vertical>
              <samples>1</samples>
              <resolution>1</resolution>
              <min_angle>0</min_angle>
              <max_angle>0</max_angle>
            </vertical>
          </scan>
          <range>
            <min>0.08</min>
            <max>10.0</max>
            <resolution>0.01</resolution>
          </range>
        </lidar>
      </sensor>
    </link>
    <joint name=\"lidar_joint\" type=\"fixed\">
      <parent>iris_with_standoffs::base_link</parent>
      <child>lidar_link</child>
    </joint>
'''
if 'lidar_link' not in sdf:
    sdf = sdf.replace('  </model>\n</sdf>', lidar + '\n  </model>\n</sdf>')
    open('$MODEL_DEST/model.sdf', 'w').write(sdf)
"

    echo "  Drone $i : $MODEL_NAME (fdm_in=$PORT_IN, fdm_out=$PORT_OUT, +LiDAR)"
done

# Générer le monde SDF : Warehouse + N drones
echo ""
echo "Génération du monde warehouse_drones.sdf ($N_DRONES drones)..."

cat > "$GENERATED_WORLD" << 'HEADER'
<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="warehouse">

    <!-- Plugins système -->
    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-imu-system"
            name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-sensors-system"
            name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-contact-system"
            name="gz::sim::systems::Contact"/>

    <!-- Physique -->
    <physics name="1ms" type="ignore">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Scène -->
    <scene>
      <ambient>0.3 0.3 0.3 1</ambient>
      <background>0.6 0.7 0.8 1</background>
    </scene>

    <!-- Soleil -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 20 0 0 0</pose>
      <diffuse>0.5 0.5 0.5 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>-0.3 0.2 -0.9</direction>
    </light>

    <!-- Éclairage intérieur -->
    <light type="point" name="light_nw">
      <pose>-7 5 5.5 0 0 0</pose>
      <diffuse>0.9 0.9 0.85 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation><range>20</range><constant>0.3</constant>
        <linear>0.01</linear><quadratic>0.001</quadratic></attenuation>
    </light>
    <light type="point" name="light_ne">
      <pose>7 5 5.5 0 0 0</pose>
      <diffuse>0.9 0.9 0.85 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation><range>20</range><constant>0.3</constant>
        <linear>0.01</linear><quadratic>0.001</quadratic></attenuation>
    </light>
    <light type="point" name="light_sw">
      <pose>-7 -5 5.5 0 0 0</pose>
      <diffuse>0.9 0.9 0.85 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation><range>20</range><constant>0.3</constant>
        <linear>0.01</linear><quadratic>0.001</quadratic></attenuation>
    </light>
    <light type="point" name="light_se">
      <pose>7 -5 5.5 0 0 0</pose>
      <diffuse>0.9 0.9 0.85 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation><range>20</range><constant>0.3</constant>
        <linear>0.01</linear><quadratic>0.001</quadratic></attenuation>
    </light>

    <!-- ============== SOL ============== -->
    <model name="floor">
      <static>true</static>
      <pose>0 0 -0.05 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>30 20 0.1</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>30 20 0.1</size></box></geometry>
          <material><ambient>0.35 0.35 0.35 1</ambient>
            <diffuse>0.45 0.45 0.45 1</diffuse></material></visual>
      </link>
    </model>

    <!-- ============== MURS ============== -->
    <model name="wall_north">
      <static>true</static><pose>0 10.1 3 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>30 0.2 6</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>30 0.2 6</size></box></geometry>
          <material><ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse></material></visual>
      </link>
    </model>
    <model name="wall_south_l">
      <static>true</static><pose>-10.5 -10.1 3 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>9 0.2 6</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>9 0.2 6</size></box></geometry>
          <material><ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse></material></visual>
      </link>
    </model>
    <model name="wall_south_r">
      <static>true</static><pose>10.5 -10.1 3 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>9 0.2 6</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>9 0.2 6</size></box></geometry>
          <material><ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse></material></visual>
      </link>
    </model>
    <model name="wall_east">
      <static>true</static><pose>15.1 0 3 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>0.2 20.4 6</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>0.2 20.4 6</size></box></geometry>
          <material><ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse></material></visual>
      </link>
    </model>
    <model name="wall_west">
      <static>true</static><pose>-15.1 0 3 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>0.2 20.4 6</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>0.2 20.4 6</size></box></geometry>
          <material><ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse></material></visual>
      </link>
    </model>

    <!-- ============== PLAFOND ============== -->
    <model name="ceiling">
      <static>true</static><pose>0 0 6.05 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>30 20 0.1</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>30 20 0.1</size></box></geometry>
          <material><ambient>0.85 0.85 0.85 1</ambient>
            <diffuse>0.9 0.9 0.9 1</diffuse></material></visual>
      </link>
    </model>

    <!-- ============== ÉTAGÈRES ============== -->
    <model name="shelf_nw">
      <static>true</static><pose>-7 5 1.5 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>6 1.2 3</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>6 1.2 3</size></box></geometry>
          <material><ambient>0.55 0.35 0.15 1</ambient>
            <diffuse>0.65 0.45 0.2 1</diffuse></material></visual>
      </link>
    </model>
    <model name="shelf_ne">
      <static>true</static><pose>7 5 1.5 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>6 1.2 3</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>6 1.2 3</size></box></geometry>
          <material><ambient>0.55 0.35 0.15 1</ambient>
            <diffuse>0.65 0.45 0.2 1</diffuse></material></visual>
      </link>
    </model>
    <model name="shelf_sw">
      <static>true</static><pose>-7 -5 1.5 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>6 1.2 3</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>6 1.2 3</size></box></geometry>
          <material><ambient>0.55 0.35 0.15 1</ambient>
            <diffuse>0.65 0.45 0.2 1</diffuse></material></visual>
      </link>
    </model>
    <model name="shelf_se">
      <static>true</static><pose>7 -5 1.5 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>6 1.2 3</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>6 1.2 3</size></box></geometry>
          <material><ambient>0.55 0.35 0.15 1</ambient>
            <diffuse>0.65 0.45 0.2 1</diffuse></material></visual>
      </link>
    </model>

    <!-- ============== CAISSES ============== -->
    <model name="crate_1">
      <static>true</static><pose>-12 -7 0.5 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>1 1 1</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>1 1 1</size></box></geometry>
          <material><ambient>0.6 0.5 0.2 1</ambient>
            <diffuse>0.7 0.6 0.3 1</diffuse></material></visual>
      </link>
    </model>
    <model name="crate_2">
      <static>true</static><pose>12 7 0.4 0 0 0.3</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>1.2 0.8 0.8</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>1.2 0.8 0.8</size></box></geometry>
          <material><ambient>0.6 0.5 0.2 1</ambient>
            <diffuse>0.7 0.6 0.3 1</diffuse></material></visual>
      </link>
    </model>

HEADER

# --- Ajouter les drones ---
START_X=$(echo "scale=2; -(($N_DRONES - 1) * $DRONE_SPACING) / 2" | bc)

echo "  Positions des drones (espacement ${DRONE_SPACING}m) :"

for i in $(seq 0 $((N_DRONES - 1))); do
    X_POS=$(echo "scale=2; $START_X + $i * $DRONE_SPACING" | bc)
    echo "    Drone $i : x=$X_POS, y=0, z=0.195"

    cat >> "$GENERATED_WORLD" << EOF

    <!-- ======== Drone $i ======== -->
    <include>
      <uri>model://iris_instance_$i</uri>
      <name>drone_$i</name>
      <pose>$X_POS 0 0.195 0 0 0</pose>
    </include>
EOF
done

cat >> "$GENERATED_WORLD" << 'FOOTER'

  </world>
</sdf>
FOOTER

echo ""
echo "==> Monde généré : $GENERATED_WORLD"

echo ""
echo "Options de rendering :"
echo "  1) Avec rendering (fenêtre Gazebo) — GPU requis"
echo "  2) Sans rendering (headless) — léger"
echo ""
read -p "Choix [1/2, défaut=2] : " RENDER_CHOICE
RENDER_CHOICE=${RENDER_CHOICE:-2}

if [ "$RENDER_CHOICE" = "2" ]; then
    GZ_HEADLESS="-s"
    echo "Mode headless activé."
else
    GZ_HEADLESS=""
fi

PIDS=()

cleanup() {
    echo ""
    echo "========================================"
    echo " Arrêt de tous les processus..."
    echo "========================================"
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    pkill -f arducopter 2>/dev/null || true
    pkill -f "gz sim" 2>/dev/null || true
    pkill -f sim_vehicle 2>/dev/null || true
    sleep 1
    echo "Nettoyage terminé."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Lancer Gazebo
echo ""
echo "==> Lancement de Gazebo avec $N_DRONES drones dans le warehouse..."
echo ""

gz sim $GZ_HEADLESS -r "$GENERATED_WORLD" &
PIDS+=($!)

echo "Attente du démarrage de Gazebo (15s)..."
sleep 15

# Lancer les instances ArduPilot SITL
echo ""
echo "==> Lancement de $N_DRONES instances ArduPilot SITL..."
echo ""

cd "$ARDUPILOT_DIR"

# Charger le profil ArduPilot
. ~/.profile 2>/dev/null || true
. ~/.bashrc 2>/dev/null || true

for i in $(seq 0 $((N_DRONES - 1))); do
    MAVLINK_PORT=$((5760 + i * 10))
  LOG_FILE="/tmp/arducopter_${i}.log"
  echo "  Drone $i : $SIM_VEHICLE_PYTHON $SIM_VEHICLE_SCRIPT -I $i → MAVLink tcp:127.0.0.1:$MAVLINK_PORT"

  "$SIM_VEHICLE_PYTHON" "$SIM_VEHICLE_SCRIPT" \
        -v ArduCopter \
        -f gazebo-iris \
        --model JSON \
        -I $i \
        --no-mavproxy \
        --no-rebuild \
    > "$LOG_FILE" 2>&1 &
  SITL_PID=$!
  PIDS+=($SITL_PID)

  sleep 5

  if ! kill -0 "$SITL_PID" 2>/dev/null; then
    echo ""
    echo "ERREUR : l'instance SITL du drone $i a quitté immédiatement."
    echo "Log : $LOG_FILE"
    tail -n 40 "$LOG_FILE" 2>/dev/null || true
    exit 1
  fi

  sleep 3
done

echo ""
echo "============================================================"
echo "  $N_DRONES DRONES ACTIFS DANS LE WAREHOUSE !"
echo "============================================================"
echo ""
echo "  Connexions MAVLink disponibles :"
echo "  ┌──────────┬──────────────────────────────┐"
echo "  │  Drone   │  Adresse MAVLink              │"
echo "  ├──────────┼──────────────────────────────┤"
for i in $(seq 0 $((N_DRONES - 1))); do
    PORT=$((5760 + i * 10))
    printf "  │  %-6s  │  tcp:127.0.0.1:%-14s │\n" "$i" "$PORT"
done
echo "  └──────────┴──────────────────────────────┘"
echo ""

wait
