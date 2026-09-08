"""
PASO 1 (version actualizada): Captura de landmarks con la nueva API de MediaPipe Tasks
----------------------------------------------------------------------------------------
MediaPipe elimino el modulo viejo "mp.solutions" (incluyendo Holistic) en sus
versiones recientes. Ahora hay que usar tres detectores separados: manos, cara
y postura corporal, cada uno con su propio modelo descargable.
 
La primera vez que corras este script, va a descargar automaticamente 3 archivos
de modelo (.task) desde los servidores oficiales de Google. Necesitas internet
la primera vez; despues quedan guardados localmente.
 
Controles:
  - Presiona 'q' para cerrar la ventana.
"""
 
import os
import urllib.request
 
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
 
# --- URLs oficiales de los modelos (Google Cloud Storage) ---
MODELOS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}
 
 
def descargar_modelos_si_faltan():
    """Descarga los modelos .task una sola vez, si no existen ya en la carpeta."""
    for nombre_archivo, url in MODELOS.items():
        if not os.path.exists(nombre_archivo):
            print(f"Descargando {nombre_archivo} ...")
            urllib.request.urlretrieve(url, nombre_archivo)
            print(f"  Listo: {nombre_archivo}")
        else:
            print(f"Ya existe: {nombre_archivo}")
 
 
def crear_detectores():
    """Crea los tres detectores en modo IMAGE (uno por frame, simple para empezar)."""
    base_hand = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
    opciones_hand = mp_vision.HandLandmarkerOptions(
        base_options=base_hand,
        num_hands=2,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    hand_detector = mp_vision.HandLandmarker.create_from_options(opciones_hand)
 
    base_face = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
    opciones_face = mp_vision.FaceLandmarkerOptions(
        base_options=base_face,
        num_faces=1,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    face_detector = mp_vision.FaceLandmarker.create_from_options(opciones_face)
 
    base_pose = mp_python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
    opciones_pose = mp_vision.PoseLandmarkerOptions(
        base_options=base_pose,
        num_poses=1,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    pose_detector = mp_vision.PoseLandmarker.create_from_options(opciones_pose)
 
    return hand_detector, face_detector, pose_detector
 
 
def dibujar_puntos(imagen, landmarks_lista, color, radio=2):
    """Dibuja circulos en cada landmark detectado. landmarks_lista es una lista
    de objetos con .x y .y normalizados (0 a 1)."""
    alto, ancho, _ = imagen.shape
    for lm in landmarks_lista:
        cx, cy = int(lm.x * ancho), int(lm.y * alto)
        cv2.circle(imagen, (cx, cy), radio, color, -1)
 
 
def main():
    descargar_modelos_si_faltan()
    hand_detector, face_detector, pose_detector = crear_detectores()
 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir la camara. Revisa que no este en uso por otra app.")
        return
 
    print("Camara iniciada. Presiona 'q' en la ventana de video para salir.")
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la camara.")
            break
 
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
        # La nueva API espera un objeto mp.Image, no un array crudo
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
 
        resultado_manos = hand_detector.detect(mp_image)
        resultado_cara = face_detector.detect(mp_image)
        resultado_pose = pose_detector.detect(mp_image)
 
        # --- Dibujar manos (verde) ---
        hay_mano_izq = False
        hay_mano_der = False
        for i, mano_landmarks in enumerate(resultado_manos.hand_landmarks):
            dibujar_puntos(frame, mano_landmarks, (0, 255, 0), radio=3)
            if i < len(resultado_manos.handedness):
                etiqueta = resultado_manos.handedness[i][0].category_name
                if etiqueta == "Left":
                    hay_mano_izq = True
                else:
                    hay_mano_der = True
 
        # --- Dibujar cara (celeste, puntos mas pequenos por ser muchos) ---
        hay_cara = False
        for cara_landmarks in resultado_cara.face_landmarks:
            dibujar_puntos(frame, cara_landmarks, (255, 255, 0), radio=1)
            hay_cara = True
 
        # --- Dibujar pose / cuerpo (rojo) ---
        hay_pose = False
        for pose_landmarks in resultado_pose.pose_landmarks:
            dibujar_puntos(frame, pose_landmarks, (0, 0, 255), radio=3)
            hay_pose = True
 
        # Texto de estado en pantalla, util para depurar
        estado = [
            f"Cara: {'SI' if hay_cara else 'NO'}",
            f"Pose: {'SI' if hay_pose else 'NO'}",
            f"Mano Izq: {'SI' if hay_mano_izq else 'NO'}",
            f"Mano Der: {'SI' if hay_mano_der else 'NO'}",
        ]
        y = 25
        for linea in estado:
            cv2.putText(frame, linea, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25
 
        cv2.imshow("Paso 1 - Deteccion de landmarks (presiona q para salir)", frame)
 
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()