#!/usr/bin/env python3
"""
QR Code System for Isaac Sim Drone Simulation
==============================================

Architecture 2 threads :
  - Thread principal (Isaac Sim) : capture les frames caméra → Queue
  - Thread daemon (décodage)     : consomme la Queue, décode avec 2 méthodes,
                                   met à jour un cache TTL

Composants :
  1. QRCodeGenerator    — génère le QR code PNG
  2. QRCodePanel        — place le QR code comme texture sur un objet 3D
  3. DroneCameraReader  — caméra Isaac Sim attachée à chaque drone
  4. QRResultCache      — cache thread-safe avec TTL
  5. decode_pyzbar / decode_opencv / decode_qr_multi — décodeurs
  6. FrameCaptureHelper — capture frames depuis le thread principal
  7. QRDecoderThread    — thread daemon de décodage
"""

from __future__ import annotations

import math
import os
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import qrcode
from PIL import Image


# ════════════════════════════════════════════════════════════════
# 1. QR Code Generator
# ════════════════════════════════════════════════════════════════

class QRCodeGenerator:
    """Génère un QR code PNG haute résolution."""

    def generate(
        self,
        data: str,
        output_path: str,
        box_size: int = 20,
        border: int = 2,
    ) -> str:
        """Génère un QR code et le sauvegarde en PNG.

        Args:
            data: texte à encoder dans le QR code.
            output_path: chemin de sauvegarde du PNG.
            box_size: taille de chaque module (pixel).
            border: largeur de la bordure (en modules).

        Returns:
            Chemin absolu du fichier PNG créé.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,  # max correction
            box_size=box_size,
            border=border,
        )
        qr.add_data(data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        img.save(output_path, "PNG")

        abs_path = os.path.abspath(output_path)
        print(f"[QR] QR code generated: '{data}' → {abs_path} "
              f"({img.size[0]}×{img.size[1]} px)")
        return abs_path


# ════════════════════════════════════════════════════════════════
# 2. QR Code Panel — objet 3D texturé dans Isaac Sim
# ════════════════════════════════════════════════════════════════

class QRCodePanel:
    """Crée un plan 3D dans la scène USD avec le QR code comme texture."""

    def __init__(
        self,
        qr_image_path: str,
        position: Tuple[float, float, float] = (0.0, 0.0, 1.8),
        size: float = 1.5,
        prim_path: str = "/World/QRCodePanel",
    ):
        self.qr_image_path = os.path.abspath(qr_image_path)
        self.position = position
        self.size = size
        self.prim_path = prim_path
        self._create()

    def _create(self):
        """Crée le mesh plan + material OmniPBR avec texture QR."""
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        stage = omni.usd.get_context().get_stage()

        # ── Mesh Plane ──
        mesh = UsdGeom.Mesh.Define(stage, self.prim_path)
        half = self.size / 2.0
        mesh.CreatePointsAttr([
            Gf.Vec3f(-half, -half, 0),
            Gf.Vec3f(half, -half, 0),
            Gf.Vec3f(half, half, 0),
            Gf.Vec3f(-half, half, 0),
        ])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)

        # UV coordinates pour mapper la texture
        texcoords = UsdGeom.PrimvarsAPI(mesh.GetPrim())
        st = texcoords.CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        )
        st.Set([
            Gf.Vec2f(0, 0), Gf.Vec2f(1, 0),
            Gf.Vec2f(1, 1), Gf.Vec2f(0, 1),
        ])

        # Position + orientation
        xf = UsdGeom.Xformable(mesh.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*self.position))
        # Panneau vertical face +Y (vers les drones) : RotateX(-90°)
        # Le mesh plat a sa normale en +Z par défaut.
        # RotateX(-90°) → normale pointe +Y = face aux drones venant de Y > pos.Y
        xf.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))

        # Make panel visible from both sides
        mesh.CreateDoubleSidedAttr(True)

        # ── Material avec texture ──
        mat_path = f"{self.prim_path}/Material"
        material = UsdShade.Material.Define(stage, mat_path)

        shader_path = f"{mat_path}/Shader"
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        # Self-lit panel (like a backlit sign) — always visible
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.8, 0.8, 0.8)
        )

        # Texture reader
        tex_path = f"{mat_path}/DiffuseTexture"
        tex_reader = UsdShade.Shader.Define(stage, tex_path)
        tex_reader.CreateIdAttr("UsdUVTexture")
        tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(self.qr_image_path)
        )
        tex_reader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex_reader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")

        # ST reader (UV coords)
        st_reader_path = f"{mat_path}/STReader"
        st_reader = UsdShade.Shader.Define(stage, st_reader_path)
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

        # Connexions
        tex_reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.ConnectableAPI(), "result"
        )
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex_reader.ConnectableAPI(), "rgb"
        )
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )

        # Bind material au mesh
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

        print(f"[QR] Panel created at {self.prim_path} "
              f"pos=({self.position[0]:.1f}, {self.position[1]:.1f}, "
              f"{self.position[2]:.1f}) size={self.size}m "
              f"texture={self.qr_image_path}")


# ════════════════════════════════════════════════════════════════
# 3. Drone Camera Reader
# ════════════════════════════════════════════════════════════════

class DroneCameraReader:
    """Caméra Isaac Sim attachée au body d'un drone.

    Capture des frames RGBA via le render product.
    IMPORTANT : get_frame() doit être appelé depuis le thread principal.
    """

    def __init__(
        self,
        drone_id: int,
        drone_prim_path: str,
        resolution: Tuple[int, int] = (640, 480),
        focal_length: float = 24.0,
    ):
        self.drone_id = drone_id
        self.cam_prim_path = f"{drone_prim_path}/body/Camera_{drone_id}"
        self.resolution = resolution
        self._annotator = None
        self._render_product = None

        self._create_camera(drone_prim_path, focal_length)

    def _create_camera(self, drone_prim_path: str, focal_length: float):
        """Crée la caméra USD sous le prim du drone."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()

        camera = UsdGeom.Camera.Define(stage, self.cam_prim_path)
        # Use wider FOV for better QR detection coverage (lower focal length)
        camera.CreateFocalLengthAttr(min(focal_length, 18.0))
        camera.CreateHorizontalApertureAttr(36.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

        # Orient camera: FORWARD along drone's Y axis (Iris body frame Y=front)
        # with a slight upward tilt to see wall-mounted QR panels.
        # USD cameras look along -Z by default.
        # Rx(+90°)  maps -Z → +Y = forward (drone nose direction).
        # Rx(+100°) adds ~10° upward tilt so panels at eye level are captured.
        xf = UsdGeom.Xformable(camera.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.15, 0.0))   # 15 cm in front (Y+)
        xf.AddRotateXYZOp().Set(Gf.Vec3f(100.0, 0.0, 0.0))  # look +Y, 10° up tilt

        print(f"[CAM] Camera created: {self.cam_prim_path} "
              f"(front-facing +Y, 10° up tilt, wide FOV)")

    def initialize(self):
        """Initialise le render product et l'annotator RGBA.
        Doit être appelé après world.reset()."""
        try:
            import omni.replicator.core as rep

            self._render_product = rep.create.render_product(
                self.cam_prim_path, self.resolution
            )
            self._annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self._annotator.attach([self._render_product])
            print(f"[CAM] Render product initialized: {self.cam_prim_path} "
                  f"({self.resolution[0]}×{self.resolution[1]})")
        except Exception as e:
            print(f"[WARN] Camera {self.drone_id} render product failed: {e}")
            self._annotator = None

    def get_frame(self) -> Optional[np.ndarray]:
        """Capture une frame RGBA. Retourne un array (H, W, 3) BGR ou None.

        DOIT être appelé depuis le thread principal Isaac Sim.
        """
        if self._annotator is None:
            return None
        try:
            data = self._annotator.get_data()
            if data is None or data.size == 0:
                return None
            # data est en RGB(A), shape (H, W, 3 ou 4)
            if data.ndim == 3 and data.shape[2] >= 3:
                # Convertir RGB → BGR pour OpenCV
                bgr = cv2.cvtColor(data[:, :, :3], cv2.COLOR_RGB2BGR)
                return bgr
            return None
        except Exception:
            return None


# ════════════════════════════════════════════════════════════════
# 4. QR Result Cache — thread-safe avec TTL
# ════════════════════════════════════════════════════════════════

class QRResultCache:
    """Stocke le dernier résultat de décodage QR avec un TTL.

    Si la prochaine frame est floue et le décodage échoue,
    on retourne le résultat du cache tant que le TTL n'est pas expiré.
    """

    def __init__(self, ttl_seconds: float = 3.0):
        self._lock = threading.Lock()
        self._data: Optional[str] = None
        self._timestamp: float = 0.0
        self._ttl = ttl_seconds
        self._hit_count = 0
        self._miss_count = 0

    def put(self, data: str):
        """Stocke un résultat de décodage réussi."""
        with self._lock:
            self._data = data
            self._timestamp = time.monotonic()

    def get(self) -> Optional[str]:
        """Retourne le résultat si le TTL n'est pas expiré, sinon None."""
        with self._lock:
            if self._data is None:
                self._miss_count += 1
                return None
            elapsed = time.monotonic() - self._timestamp
            if elapsed <= self._ttl:
                self._hit_count += 1
                return self._data
            self._miss_count += 1
            return None

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cached_data": self._data,
                "ttl": self._ttl,
                "hits": self._hit_count,
                "misses": self._miss_count,
                "age_seconds": round(time.monotonic() - self._timestamp, 2)
                if self._data else None,
            }


# ════════════════════════════════════════════════════════════════
# 5. QR Decoders — 2 méthodes de décodage
# ════════════════════════════════════════════════════════════════

def decode_pyzbar(image: np.ndarray) -> Optional[str]:
    """Décode avec pyzbar (wraps libzbar). Très robuste au flou."""
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        results = pyzbar_decode(image)
        if results:
            return results[0].data.decode("utf-8", errors="replace")
    except Exception:
        pass
    return None


def decode_opencv(image: np.ndarray) -> Optional[str]:
    """Décode avec cv2.QRCodeDetector. Pas de dépendance externe."""
    try:
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(image)
        if data:
            return data
    except Exception:
        pass
    return None


def decode_qr_multi(image: np.ndarray) -> Tuple[Optional[str], str]:
    """Essaie les 2 méthodes de décodage.

    Returns:
        (data, method) — data est le texte décodé ou None,
        method est le nom de la méthode qui a réussi.
    """
    # Méthode 1 : pyzbar (la plus robuste)
    result = decode_pyzbar(image)
    if result is not None:
        return result, "pyzbar"

    # Méthode 2 : OpenCV QRCodeDetector
    result = decode_opencv(image)
    if result is not None:
        return result, "opencv"

    return None, "none"


# ════════════════════════════════════════════════════════════════
# 6. Frame Info — données partagées entre threads
# ════════════════════════════════════════════════════════════════

@dataclass
class FrameInfo:
    """Données d'une frame capturée, passées du thread principal au décodeur."""
    drone_id: int
    tick: int
    image: np.ndarray       # BGR, shape (H, W, 3)
    saved_path: str          # chemin du PNG sauvegardé
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class DroneDetectionRecord:
    """Résultat de détection QR pour un drone spécifique."""
    drone_id: int
    last_status: str = "none"         # "success" | "failed" | "cached" | "none"
    last_data: Optional[str] = None
    last_method: str = "none"
    last_frame_path: str = ""
    last_annotated_path: str = ""
    decoded_count: int = 0
    failed_count: int = 0
    total_count: int = 0
    last_timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drone_id": self.drone_id,
            "last_status": self.last_status,
            "last_data": self.last_data,
            "last_method": self.last_method,
            "last_frame_path": os.path.basename(self.last_frame_path) if self.last_frame_path else "",
            "last_annotated_path": os.path.basename(self.last_annotated_path) if self.last_annotated_path else "",
            "decoded_count": self.decoded_count,
            "failed_count": self.failed_count,
            "total_count": self.total_count,
            "success_rate": round(self.decoded_count / max(self.total_count, 1) * 100, 1),
        }


# ════════════════════════════════════════════════════════════════
# 7. Frame Capture Helper — appelé depuis le thread principal
# ════════════════════════════════════════════════════════════════

class FrameCaptureHelper:
    """Capture les frames des caméras et les pousse dans une queue.

    IMPORTANT : capture_once() est appelé depuis le thread principal
    Isaac Sim car les APIs de rendu ne sont pas thread-safe.
    """

    def __init__(
        self,
        cameras: List[DroneCameraReader],
        frame_queue: queue.Queue,
        output_dir: str,
        capture_interval: int = 5,
    ):
        self.cameras = cameras
        self.frame_queue = frame_queue
        self.output_dir = output_dir
        self.capture_interval = capture_interval
        self._tick = 0
        self._frame_count = 0

        # Créer les sous-dossiers par drone
        for cam in cameras:
            drone_dir = os.path.join(output_dir, f"drone_{cam.drone_id}")
            os.makedirs(drone_dir, exist_ok=True)

    def tick(self):
        """Appelé à chaque physics tick. Capture si l'intervalle est atteint."""
        self._tick += 1
        if self._tick % self.capture_interval == 0:
            self.capture_once()

    def capture_once(self):
        """Capture une frame de chaque caméra, sauvegarde et enqueue."""
        for cam in self.cameras:
            frame = cam.get_frame()
            if frame is None:
                continue

            # Sauvegarder comme PNG
            drone_dir = os.path.join(self.output_dir, f"drone_{cam.drone_id}")
            filename = f"frame_{self._frame_count:06d}.png"
            filepath = os.path.join(drone_dir, filename)

            cv2.imwrite(filepath, frame)

            # Sauvegarder aussi la frame la plus récente sous un nom fixe
            # pour que le dashboard puisse toujours la lire
            latest_path = os.path.join(self.output_dir, f"latest_drone_{cam.drone_id}.jpg")
            cv2.imwrite(latest_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            # Pousser dans la queue (copie pour thread-safety)
            info = FrameInfo(
                drone_id=cam.drone_id,
                tick=self._tick,
                image=frame.copy(),
                saved_path=filepath,
            )

            try:
                self.frame_queue.put_nowait(info)
            except queue.Full:
                # Queue pleine, on drop la frame la plus ancienne
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait(info)

        self._frame_count += 1

        if self._frame_count <= 3 or self._frame_count % 100 == 0:
            print(f"[CAM] Captured frame #{self._frame_count} "
                  f"(tick={self._tick}, {len(self.cameras)} cameras)")


# ════════════════════════════════════════════════════════════════
# 8. QR Decoder Thread — Thread 2 (daemon)
# ════════════════════════════════════════════════════════════════

class QRDecoderThread(threading.Thread):
    """Thread daemon qui consomme les frames et décode les QR codes.

    Utilise decode_qr_multi (2 méthodes) pour la robustesse.
    Met à jour le QRResultCache à chaque décodage réussi.
    Si le décodage échoue, le cache fournit le dernier résultat valide.

    Sauvegarde aussi une frame annotée (bounding box vert/rouge) pour le dashboard.
    """

    def __init__(
        self,
        frame_queue: queue.Queue,
        cache: QRResultCache,
        output_dir: str = "/tmp",
        drone_ids: Optional[List[int]] = None,
    ):
        super().__init__(daemon=True, name="QRDecoderThread")
        self.frame_queue = frame_queue
        self.cache = cache
        self.output_dir = output_dir
        self._stop_event = threading.Event()
        self._decoded_count = 0
        self._failed_count = 0
        self._total_count = 0
        # Per-drone detection tracking
        self._lock = threading.Lock()
        self._drone_records: Dict[int, DroneDetectionRecord] = {}
        if drone_ids:
            for did in drone_ids:
                self._drone_records[did] = DroneDetectionRecord(drone_id=did)
        # Detection history for charts
        self._detection_history: List[Dict[str, Any]] = []

    def _annotate_and_save(self, image: np.ndarray, drone_id: int,
                           success: bool, data: Optional[str],
                           method: str) -> str:
        """Dessine un bounding box sur la frame et la sauvegarde.
        Vert = QR détecté, Rouge = échec."""
        annotated = image.copy()
        h, w = annotated.shape[:2]

        if success:
            # Essayer de trouver le QR pour dessiner le bbox
            try:
                from pyzbar.pyzbar import decode as pyzbar_decode
                results = pyzbar_decode(image)
                if results:
                    for r in results:
                        pts = r.polygon
                        if pts and len(pts) >= 4:
                            points = np.array([(p.x, p.y) for p in pts], dtype=np.int32)
                            cv2.polylines(annotated, [points], True, (0, 255, 0), 3)
                        else:
                            x, y, bw, bh = r.rect
                            cv2.rectangle(annotated, (x, y), (x+bw, y+bh), (0, 255, 0), 3)
            except Exception:
                pass
            # Label
            cv2.putText(annotated, f"QR: {data}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Method: {method}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # Bordure rouge
            cv2.rectangle(annotated, (5, 5), (w-5, h-5), (0, 0, 255), 3)
            cached = self.cache.get()
            label = f"CACHE: {cached}" if cached else "NO QR DETECTED"
            cv2.putText(annotated, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Status banner
        status_color = (0, 180, 0) if success else (0, 0, 200)
        cv2.rectangle(annotated, (0, h-35), (w, h), status_color, -1)
        status_text = "DECODED" if success else "FAILED"
        cv2.putText(annotated, f"D{drone_id} | {status_text}",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_path = os.path.join(self.output_dir, "camera_frames",
                                f"annotated_drone_{drone_id}.jpg")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return out_path

    def run(self):
        """Boucle principale du thread de décodage."""
        print("[QR-DECODER] Thread started")
        while not self._stop_event.is_set():
            try:
                info = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self._total_count += 1
            data, method = decode_qr_multi(info.image)

            # Ensure drone record exists
            with self._lock:
                if info.drone_id not in self._drone_records:
                    self._drone_records[info.drone_id] = DroneDetectionRecord(
                        drone_id=info.drone_id)
                rec = self._drone_records[info.drone_id]
                rec.total_count += 1
                rec.last_frame_path = info.saved_path
                rec.last_timestamp = time.time()

            if data is not None:
                self.cache.put(data)
                self._decoded_count += 1

                # Annotate and save
                ann_path = self._annotate_and_save(
                    info.image, info.drone_id, True, data, method)

                with self._lock:
                    rec.last_status = "success"
                    rec.last_data = data
                    rec.last_method = method
                    rec.decoded_count += 1
                    rec.last_annotated_path = ann_path

                if self._decoded_count <= 5 or self._decoded_count % 50 == 0:
                    print(f"[QR-DECODER] ✓ Decoded from drone {info.drone_id} "
                          f"frame {info.saved_path}: '{data}' "
                          f"(method={method}, "
                          f"total={self._decoded_count}/{self._total_count})")
            else:
                self._failed_count += 1
                cached = self.cache.get()
                status = "cached" if cached else "failed"

                # Annotate and save
                ann_path = self._annotate_and_save(
                    info.image, info.drone_id, False, None, "none")

                with self._lock:
                    rec.last_status = status
                    rec.last_data = cached
                    rec.last_method = "cache" if cached else "none"
                    rec.failed_count += 1
                    rec.last_annotated_path = ann_path

                if self._failed_count <= 3 or self._failed_count % 50 == 0:
                    tag = f"cache: '{cached}'" if cached else "no cache"
                    print(f"[QR-DECODER] ✗ Decode failed drone {info.drone_id}, "
                          f"{tag} (fails={self._failed_count}/{self._total_count})")

            # Add to detection history (for chart)
            with self._lock:
                self._detection_history.append({
                    "tick": info.tick,
                    "drone_id": info.drone_id,
                    "success": data is not None,
                    "method": method if data else "none",
                    "cached": data is None and cached is not None
                              if 'cached' in dir() else False,
                })
                # Keep last 500 entries
                if len(self._detection_history) > 500:
                    self._detection_history = self._detection_history[-500:]

            # Write qr_state.json for dashboard
            self._write_state_json()

            self.frame_queue.task_done()

        print(f"[QR-DECODER] Thread stopped "
              f"(decoded={self._decoded_count}, "
              f"failed={self._failed_count}, "
              f"total={self._total_count})")

    def _write_state_json(self):
        """Write QR detection state to JSON for the dashboard."""
        try:
            with self._lock:
                state = {
                    "timestamp": time.time(),
                    "total_decoded": self._decoded_count,
                    "total_failed": self._failed_count,
                    "total_processed": self._total_count,
                    "success_rate": round(
                        self._decoded_count / max(self._total_count, 1) * 100, 1),
                    "cache": self.cache.stats(),
                    "drones": {str(k): v.to_dict()
                               for k, v in self._drone_records.items()},
                    "history": self._detection_history[-100:],
                }
            path = os.path.join(self.output_dir, "qr_state.json")
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f, separators=(",", ":"))
            os.replace(tmp, path)
        except Exception:
            pass  # non-critical

    def stop(self):
        """Arrête proprement le thread."""
        self._stop_event.set()
        self.join(timeout=5.0)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "decoded": self._decoded_count,
                "failed": self._failed_count,
                "total": self._total_count,
                "queue_size": self.frame_queue.qsize(),
                "cache": self.cache.stats(),
                "drones": {str(k): v.to_dict()
                           for k, v in self._drone_records.items()},
            }


# ════════════════════════════════════════════════════════════════
# 9. Convenience — setup complet pour le main
# ════════════════════════════════════════════════════════════════

def setup_qr_system(
    qr_data: str,
    output_dir: str,
    drone_prim_paths: List[str],
    drone_ids: List[int],
    panel_position: Tuple[float, float, float] = (0.0, 0.0, 1.8),
    panel_size: float = 1.5,
    cache_ttl: float = 3.0,
    capture_interval: int = 5,
    camera_resolution: Tuple[int, int] = (640, 480),
) -> Dict[str, Any]:
    """Setup complet du système QR code.

    Appelé dans main() après setup_world() et avant world.reset().
    Retourne un dict avec tous les composants créés.

    Args:
        qr_data: texte à encoder dans le QR code.
        output_dir: dossier de sortie pour le QR code et les frames.
        drone_prim_paths: liste des prim paths des drones.
        drone_ids: liste des IDs des drones.
        panel_position: position 3D du panneau QR dans la scène.
        panel_size: taille du panneau en mètres.
        cache_ttl: durée du cache en secondes.
        capture_interval: intervalle de capture en physics ticks.
        camera_resolution: résolution des caméras (W, H).

    Returns:
        Dict avec les clés: qr_generator, qr_panel, cameras,
        frame_queue, cache, capture_helper, decoder_thread.
    """
    # 1. Générer le QR code
    gen = QRCodeGenerator()
    qr_image_path = gen.generate(
        qr_data, os.path.join(output_dir, "qr_code.png")
    )

    # 2. Placer le panneau dans la scène
    panel = QRCodePanel(qr_image_path, panel_position, panel_size)

    # 3. Créer les caméras
    cameras = []
    for drone_id, prim_path in zip(drone_ids, drone_prim_paths):
        cam = DroneCameraReader(
            drone_id, prim_path, camera_resolution
        )
        cameras.append(cam)

    # 4. Queue + Cache + Threads
    frame_queue = queue.Queue(maxsize=50)
    cache = QRResultCache(ttl_seconds=cache_ttl)

    frame_dir = os.path.join(output_dir, "camera_frames")
    capture_helper = FrameCaptureHelper(
        cameras, frame_queue, frame_dir, capture_interval
    )

    decoder_thread = QRDecoderThread(
        frame_queue, cache,
        output_dir=output_dir,
        drone_ids=drone_ids,
    )

    return {
        "qr_generator": gen,
        "qr_image_path": qr_image_path,
        "qr_panel": panel,
        "cameras": cameras,
        "frame_queue": frame_queue,
        "cache": cache,
        "capture_helper": capture_helper,
        "decoder_thread": decoder_thread,
    }


def initialize_cameras(cameras: List[DroneCameraReader]):
    """Initialise les render products. Appelé après world.reset()."""
    for cam in cameras:
        cam.initialize()
    print(f"[QR] {len(cameras)} cameras initialized")
