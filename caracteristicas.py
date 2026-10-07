"""
CARACTERISTICAS COMPARTIDAS (usado por Grabacion.py y reconocer.py)
------------------------------------------------------------------------
Aqui vive TODO el codigo que convierte lo que ve la camara en el vector de
entrada del modelo. Grabacion y reconocimiento en vivo importan este mismo
archivo, asi las caracteristicas son identicas en ambos y no hay desajuste
entre lo que el modelo aprendio y lo que ve en vivo.

Sistema de referencia corporal:
  - origen  = punto medio de los hombros
  - unidad  = ancho de hombros
  - lineas horizontales: cabeza (nariz), hombros, cintura
  - linea vertical: eje del cuerpo
  - Si no se ven los hombros se estiman con el tamaño de la cara; si no se
    ven las caderas, la cintura se estima bajo los hombros.

Muestreo temporal fijo:
  - Cada secuencia son FRAMES_POR_SECUENCIA muestras tomadas a FPS_SECUENCIA
    muestras por segundo (por reloj real), no "a la velocidad que de la PC".
    Con 30 muestras a 15 por segundo, cada secuencia dura 2 segundos y las
    velocidades significan lo mismo al grabar y al reconocer.
"""

import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ============================================================
# CONFIGURACION GENERAL
# ============================================================
FRAMES_POR_SECUENCIA = 30
FPS_SECUENCIA = 15          # muestras por segundo (30 muestras = 2 segundos)

# Tamaños de los vectores (se verifican en preparar.py)
NUM_FEATURES_MANO = 2 + 2 + 4 + 3 + 1 + 1 + 63           # 76
NUM_FEATURES_GLOBALES = 5
NUM_FEATURES_FRAME = 2 * NUM_FEATURES_MANO + NUM_FEATURES_GLOBALES   # 157
DIM_RESUMEN_MANO = FRAMES_POR_SECUENCIA * 2 + 2 + 1 + 4 + 4 + 3
DIM_TRAYECTORIA = 2 * DIM_RESUMEN_MANO + FRAMES_POR_SECUENCIA          # 178

# Parametros del sistema de referencia corporal
VIS_MIN = 0.5
FACTOR_ESCALA_CARA = 2.5
FACTOR_NARIZ_HOMBROS = 0.55
FACTOR_CINTURA = 1.3
X_CENTRO = 0.35
SUAVIZADO_REF = 0.6

NOMBRES_ZONA_V = ["sobre cabeza", "cabeza-hombros", "hombros-cintura", "bajo cintura"]
NOMBRES_ZONA_H = ["izq", "centro", "der"]
COLOR_MANO = {"izq": (0, 255, 0), "der": (0, 165, 255)}

# Normalizacion
UMBRAL_STD = 1e-3           # columnas casi constantes: no se escalan (std = 1)
CLIP_NORMALIZADO = 10.0     # recorta valores extremos tras normalizar

MODELOS_MEDIAPIPE = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


# ============================================================
# RELOJ Y MUESTREO
# ============================================================
class Reloj:
    """Timestamps estrictamente crecientes en ms (necesario para RunningMode.VIDEO)."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.ultimo = 0

    def ahora_ms(self):
        t = int((time.monotonic() - self.t0) * 1000)
        if t <= self.ultimo:
            t = self.ultimo + 1
        self.ultimo = t
        return t


class Muestreador:
    """Decide en que frames de camara se toma una muestra, para lograr
    FPS_SECUENCIA muestras por segundo de reloj real."""

    def __init__(self, fps=FPS_SECUENCIA):
        self.intervalo = 1.0 / fps
        self.proximo = 0.0

    def reiniciar(self):
        self.proximo = 0.0

    def toca(self):
        ahora = time.monotonic()
        if ahora >= self.proximo:
            self.proximo = max(self.proximo + self.intervalo, ahora)
            return True
        return False


# ============================================================
# DETECTORES MEDIAPIPE
# ============================================================
def descargar_modelos_si_faltan():
    for nombre_archivo, url in MODELOS_MEDIAPIPE.items():
        if not os.path.exists(nombre_archivo):
            print(f"Descargando {nombre_archivo} ...")
            urllib.request.urlretrieve(url, nombre_archivo)


def crear_detectores():
    base_hand = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
    opciones_hand = mp_vision.HandLandmarkerOptions(
        base_options=base_hand,
        num_hands=2,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    hand_detector = mp_vision.HandLandmarker.create_from_options(opciones_hand)

    base_face = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
    opciones_face = mp_vision.FaceLandmarkerOptions(
        base_options=base_face, num_faces=1, running_mode=mp_vision.RunningMode.VIDEO,
    )
    face_detector = mp_vision.FaceLandmarker.create_from_options(opciones_face)

    base_pose = mp_python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
    opciones_pose = mp_vision.PoseLandmarkerOptions(
        base_options=base_pose, num_poses=1, running_mode=mp_vision.RunningMode.VIDEO,
    )
    pose_detector = mp_vision.PoseLandmarker.create_from_options(opciones_pose)

    return hand_detector, face_detector, pose_detector


def detectar_todo(frame_bgr, hand_detector, face_detector, pose_detector, timestamp_ms):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    r_manos = hand_detector.detect_for_video(mp_image, timestamp_ms)
    r_cara = face_detector.detect_for_video(mp_image, timestamp_ms)
    r_pose = pose_detector.detect_for_video(mp_image, timestamp_ms)
    return r_manos, r_cara, r_pose


# ============================================================
# EXTRACCION POR FRAME: manos y referencia corporal (en pixeles)
# ============================================================
def extraer_manos(r_manos, ancho, alto):
    """Devuelve {'izq': array(21,3) o None, 'der': array(21,3) o None} en pixeles."""
    manos = {"izq": None, "der": None}
    for i, landmarks in enumerate(r_manos.hand_landmarks):
        arr = np.array([[lm.x * ancho, lm.y * alto, lm.z * ancho] for lm in landmarks])
        etiqueta = "izq"
        if i < len(r_manos.handedness):
            etiqueta = "izq" if r_manos.handedness[i][0].category_name == "Left" else "der"
        if manos[etiqueta] is not None:  # etiqueta repetida: usar el otro hueco
            etiqueta = "der" if etiqueta == "izq" else "izq"
        if manos[etiqueta] is None:
            manos[etiqueta] = arr
    return manos


def _visible(lm):
    return (getattr(lm, "visibility", None) or 0.0) >= VIS_MIN


def calcular_referencia(r_cara, r_pose, ancho, alto, ref_previa):
    """Calcula origen, escala y lineas horizontales del cuerpo (en pixeles).
    Usa puntos reales si estan visibles y estimados si no. Si no hay nada
    utilizable, conserva la referencia anterior."""
    pose = r_pose.pose_landmarks[0] if r_pose.pose_landmarks else None
    cara = r_cara.face_landmarks[0] if r_cara.face_landmarks else None

    def px(lm):
        return np.array([lm.x * ancho, lm.y * alto])

    nariz = None
    if cara is not None:
        nariz = px(cara[1])
    elif pose is not None and _visible(pose[0]):
        nariz = px(pose[0])

    hombros_ok = pose is not None and _visible(pose[11]) and _visible(pose[12])

    if hombros_ok:
        h_izq, h_der = px(pose[11]), px(pose[12])
        origen = (h_izq + h_der) / 2
        escala = float(np.linalg.norm(h_izq - h_der))
        hombros_real = True
        if nariz is None:
            nariz = px(pose[0])
    elif nariz is not None and cara is not None:
        escala = FACTOR_ESCALA_CARA * float(np.linalg.norm(px(cara[234]) - px(cara[454])))
        origen = nariz + np.array([0.0, FACTOR_NARIZ_HOMBROS * escala])
        hombros_real = False
    else:
        return ref_previa

    if escala < 1.0:
        return ref_previa

    y_cintura = origen[1] + FACTOR_CINTURA * escala
    cintura_real = False
    if pose is not None and _visible(pose[23]) and _visible(pose[24]):
        y_caderas = (pose[23].y + pose[24].y) / 2 * alto
        if y_caderas > origen[1] + 0.3 * escala:
            y_cintura = y_caderas
            cintura_real = True

    ref = {
        "origen": np.array(origen, dtype=float),
        "escala": float(escala),
        "nariz": np.array(nariz, dtype=float),
        "y_cabeza": float(nariz[1]),
        "y_hombros": float(origen[1]),
        "y_cintura": float(y_cintura),
        "hombros_real": hombros_real,
        "cintura_real": cintura_real,
    }

    if ref_previa is not None:
        w = SUAVIZADO_REF
        for k in ("origen", "escala", "nariz", "y_cabeza", "y_hombros", "y_cintura"):
            ref[k] = w * ref[k] + (1 - w) * ref_previa[k]
    return ref


def leer_y_detectar(cap, detectores, reloj, ref):
    """Lee un frame, lo voltea (espejo), detecta y actualiza la referencia.
    Devuelve (frame, manos, ref) o None si la camara fallo."""
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.flip(frame, 1)
    alto, ancho = frame.shape[:2]
    r_m, r_c, r_p = detectar_todo(frame, *detectores, reloj.ahora_ms())
    manos = extraer_manos(r_m, ancho, alto)
    ref = calcular_referencia(r_c, r_p, ancho, alto, ref)
    return frame, manos, ref


def a_cuerpo(punto_xy, ref):
    """Pixeles -> coordenadas del cuerpo (origen = hombros, unidad = ancho de hombros)."""
    return (np.asarray(punto_xy[:2], dtype=float) - ref["origen"]) / ref["escala"]


def zona_vertical(y, y_cabeza_n, y_cintura_n):
    if y < y_cabeza_n:
        return 0
    if y < 0.0:  # linea de hombros = 0
        return 1
    if y < y_cintura_n:
        return 2
    return 3


def zona_horizontal(x):
    if x < -X_CENTRO:
        return 0
    if x > X_CENTRO:
        return 2
    return 1


# ============================================================
# CONSTRUCCION DE LA SECUENCIA Y DEL VECTOR DE TRAYECTORIA
# ============================================================
def construir_secuencia(raws):
    """raws: lista (una por muestra) de {'manos':..., 'ref':...} en pixeles.
    Devuelve (secuencia, trayectoria, zonas, info) o None si no hay referencia."""
    T = len(raws)
    refs = [r["ref"] for r in raws]
    primero = next((r for r in refs if r is not None), None)
    if primero is None:
        return None
    refs_ok, actual = [], primero
    for r in refs:
        if r is not None:
            actual = r
        refs_ok.append(actual)

    yc = np.array([(r["y_cabeza"] - r["origen"][1]) / r["escala"] for r in refs_ok])
    yw = np.array([(r["y_cintura"] - r["origen"][1]) / r["escala"] for r in refs_ok])
    nariz_n = np.array([a_cuerpo(r["nariz"], r) for r in refs_ok])
    hombros_real = np.array([r["hombros_real"] for r in refs_ok], dtype=float)
    cintura_real = np.array([r["cintura_real"] for r in refs_ok], dtype=float)

    def procesar_mano(nombre):
        presente = np.array([r["manos"][nombre] is not None for r in raws], dtype=float)
        idx = np.where(presente > 0)[0]
        pos = np.zeros((T, 2))
        forma = np.zeros((T, 63))
        zy = np.zeros(T, dtype=int)
        zx = np.zeros(T, dtype=int)
        hay = len(idx) > 0
        if hay:
            muñecas = np.array([a_cuerpo(raws[i]["manos"][nombre][0], refs_ok[i]) for i in idx])
            t_all = np.arange(T)
            for d in range(2):
                # Huecos sin deteccion: se rellenan uniendo los puntos vecinos
                pos[:, d] = np.interp(t_all, idx, muñecas[:, d])
            for i in idx:
                lm = raws[i]["manos"][nombre]
                tam = np.linalg.norm(lm[9, :2] - lm[0, :2]) + 1e-6
                forma[i] = ((lm - lm[0]) / tam).flatten()
            zy = np.array([zona_vertical(pos[t, 1], yc[t], yw[t]) for t in range(T)])
            zx = np.array([zona_horizontal(pos[t, 0]) for t in range(T)])
        vel = np.gradient(pos, axis=0) if (hay and T > 1) else np.zeros((T, 2))
        dist_nariz = np.linalg.norm(pos - nariz_n, axis=1) if hay else np.zeros(T)
        oh_y = np.eye(4)[zy] * (1.0 if hay else 0.0)
        oh_x = np.eye(3)[zx] * (1.0 if hay else 0.0)
        feat = np.concatenate(
            [pos, vel, oh_y, oh_x, dist_nariz[:, None], presente[:, None], forma], axis=1
        )
        return {"pos": pos, "zy": zy, "zx": zx, "hay": hay, "feat": feat, "presente": presente}

    izq = procesar_mano("izq")
    der = procesar_mano("der")

    if izq["hay"] and der["hay"]:
        dist_manos = np.linalg.norm(izq["pos"] - der["pos"], axis=1)
    else:
        dist_manos = np.zeros(T)

    globales = np.stack([yc, yw, hombros_real, cintura_real, dist_manos], axis=1)
    secuencia = np.concatenate([izq["feat"], der["feat"], globales], axis=1).astype(np.float32)

    def resumen(m):
        if not m["hay"]:
            return np.zeros(T * 2 + 2 + 1 + 4 + 4 + 3)
        pos = m["pos"]
        desplaz = pos[-1] - pos[0]
        longitud = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
        cajas = np.concatenate([pos.min(axis=0), pos.max(axis=0)])
        tz_v = np.bincount(m["zy"], minlength=4) / T
        tz_h = np.bincount(m["zx"], minlength=3) / T
        return np.concatenate([pos.flatten(), desplaz, [longitud], cajas, tz_v, tz_h])

    trayectoria = np.concatenate([resumen(izq), resumen(der), dist_manos]).astype(np.float32)
    zonas = np.stack([izq["zy"], izq["zx"], der["zy"], der["zx"]], axis=1).astype(np.int8)

    info = {
        "frames_sin_mano": int(np.sum((izq["presente"] + der["presente"]) == 0)),
        "frac_hombros_reales": float(hombros_real.mean()),
        "frac_cintura_real": float(cintura_real.mean()),
    }
    return secuencia, trayectoria, zonas, info


# ============================================================
# NORMALIZACION (la usan preparar.py y reconocer.py)
# ============================================================
def calcular_estadisticas(x, ejes):
    """Media y desviacion por caracteristica. Las columnas casi constantes
    (std < UMBRAL_STD) reciben std = 1 para no amplificar cambios minimos."""
    media = x.mean(axis=ejes)
    desviacion = x.std(axis=ejes)
    desviacion = np.where(desviacion < UMBRAL_STD, 1.0, desviacion)
    return media.astype(np.float32), desviacion.astype(np.float32)


def normalizar(x, media, desviacion):
    y = (x - media) / desviacion
    return np.clip(y, -CLIP_NORMALIZADO, CLIP_NORMALIZADO).astype(np.float32)


# ============================================================
# DIBUJO EN VIVO
# ============================================================
def _linea_punteada(img, p1, p2, color, paso=10):
    x1, y1 = p1
    x2, y2 = p2
    largo = int(np.hypot(x2 - x1, y2 - y1))
    for d in range(0, largo, paso * 2):
        a = d / max(largo, 1)
        b = min(d + paso, largo) / max(largo, 1)
        cv2.line(img, (int(x1 + (x2 - x1) * a), int(y1 + (y2 - y1) * a)),
                 (int(x1 + (x2 - x1) * b), int(y1 + (y2 - y1) * b)), color, 1)


def dibujar_overlay(frame, manos, ref, estelas=None):
    alto, ancho = frame.shape[:2]
    if ref is not None:
        lineas = [
            ("cabeza", ref["y_cabeza"], (0, 255, 255), True),
            ("hombros", ref["y_hombros"], (255, 255, 0), ref["hombros_real"]),
            ("cintura", ref["y_cintura"], (255, 0, 255), ref["cintura_real"]),
        ]
        for nombre, y, color, real in lineas:
            y = int(y)
            if real:
                cv2.line(frame, (0, y), (ancho, y), color, 1)
            else:
                _linea_punteada(frame, (0, y), (ancho, y), color)
            cv2.putText(frame, nombre + ("" if real else " (est)"), (5, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        ox = int(ref["origen"][0])
        cv2.line(frame, (ox, 0), (ox, alto), (200, 200, 200), 1)
        cv2.circle(frame, (int(ref["nariz"][0]), int(ref["nariz"][1])), 5, (0, 255, 255), -1)
        cv2.circle(frame, (ox, int(ref["origen"][1])), 5, (255, 255, 0), -1)

    if estelas:
        for nombre, puntos in estelas.items():
            if len(puntos) > 1:
                cv2.polylines(frame, [np.array(puntos, dtype=np.int32)], False,
                              COLOR_MANO[nombre], 2)

    for nombre in ("izq", "der"):
        lm = manos[nombre]
        if lm is None:
            continue
        for p in lm:
            cv2.circle(frame, (int(p[0]), int(p[1])), 2, COLOR_MANO[nombre], -1)
        cv2.circle(frame, (int(lm[0][0]), int(lm[0][1])), 8, COLOR_MANO[nombre], 2)


def describir_zonas(manos, ref):
    partes = []
    for nombre in ("izq", "der"):
        lm = manos[nombre]
        if lm is None or ref is None:
            partes.append(f"{nombre}: --")
            continue
        x, y = a_cuerpo(lm[0], ref)
        yc = (ref["y_cabeza"] - ref["origen"][1]) / ref["escala"]
        yw = (ref["y_cintura"] - ref["origen"][1]) / ref["escala"]
        partes.append(f"{nombre}: {NOMBRES_ZONA_V[zona_vertical(y, yc, yw)]} / "
                      f"{NOMBRES_ZONA_H[zona_horizontal(x)]}")
    return "  |  ".join(partes)
