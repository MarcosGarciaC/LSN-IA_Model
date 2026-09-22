"""
PASO 1 : Grabacion de dataset con menu de navegacion
--------------------------------------------------------------------------
Flujo:
  1. Se muestra un menu en la terminal con todas las SEÑAS y cuantas
     repeticiones llevas de cada una.
  2. Escribes el numero de la seña que quieres grabar y presionas Enter.
  3. Se abre la camara mostrando los landmarks en vivo.
  4. Presiona ESPACIO para grabar una repeticion (cuenta regresiva + 30 frames).
  5. Presiona ESC para volver al menu y elegir otra seña.
  6. Escribe 'q' en el menu para salir del programa.
  comando para activar el entorno venv\\Scripts\\activate

  CAMBIO: ahora usa RunningMode.VIDEO en vez de IMAGE, para que el detector
  tenga memoria entre frames (tracking). Esto ayuda cuando las manos se
  superponen momentaneamente, como en senas donde se juntan (ej: por_favor).
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
# CONFIGURACION - agrega aqui todas tus SEÑAS, las que quieras
# ============================================================
SEÑAS = [
    "hola", "gracias", "por_favor", "adios", "buenos_dias",
    "J", "nada", "reposo"  # ejemplo de seña dinamica + clases neutrales
    # aca se deben agregar mas señas que es la parte mas larga del proyecto
]

CARPETA_DATASET = "dataset"
FRAMES_POR_SECUENCIA = 30
REPETICIONES_OBJETIVO = 25

MODELOS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

N_LANDMARKS_MANO = 21
N_LANDMARKS_CARA = 478
N_LANDMARKS_POSE = 33


def descargar_modelos_si_faltan():
    for nombre_archivo, url in MODELOS.items():
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


def landmarks_a_vector(resultado_manos, resultado_cara, resultado_pose):
    mano_izq = np.zeros(N_LANDMARKS_MANO * 3)
    mano_der = np.zeros(N_LANDMARKS_MANO * 3)
    for i, mano_landmarks in enumerate(resultado_manos.hand_landmarks):
        vector = np.array([[lm.x, lm.y, lm.z] for lm in mano_landmarks]).flatten()
        if i < len(resultado_manos.handedness):
            etiqueta = resultado_manos.handedness[i][0].category_name
            if etiqueta == "Left":
                mano_izq = vector
            else:
                mano_der = vector

    cara = np.zeros(N_LANDMARKS_CARA * 3)
    if resultado_cara.face_landmarks:
        cara_landmarks = resultado_cara.face_landmarks[0]
        cara = np.array([[lm.x, lm.y, lm.z] for lm in cara_landmarks]).flatten()

    pose = np.zeros(N_LANDMARKS_POSE * 4)
    if resultado_pose.pose_landmarks:
        pose_landmarks = resultado_pose.pose_landmarks[0]
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_landmarks]
        ).flatten()

    return np.concatenate([mano_izq, mano_der, cara, pose])


def dibujar_puntos(imagen, landmarks_lista, color, radio=2):
    alto, ancho, _ = imagen.shape
    for lm in landmarks_lista:
        cx, cy = int(lm.x * ancho), int(lm.y * alto)
        cv2.circle(imagen, (cx, cy), radio, color, -1)


def dibujar_deteccion_en_frame(frame, r_manos, r_cara, r_pose):
    hay_mano_izq, hay_mano_der = False, False
    for i, mano_landmarks in enumerate(r_manos.hand_landmarks):
        dibujar_puntos(frame, mano_landmarks, (0, 255, 0), radio=3)
        if i < len(r_manos.handedness):
            etiqueta = r_manos.handedness[i][0].category_name
            if etiqueta == "Left":
                hay_mano_izq = True
            else:
                hay_mano_der = True

    hay_cara = False
    for cara_landmarks in r_cara.face_landmarks:
        dibujar_puntos(frame, cara_landmarks, (255, 255, 0), radio=1)
        hay_cara = True

    hay_pose = False
    for pose_landmarks in r_pose.pose_landmarks:
        dibujar_puntos(frame, pose_landmarks, (0, 0, 255), radio=3)
        hay_pose = True

    return hay_mano_izq, hay_mano_der, hay_cara, hay_pose


def contar_repeticiones_existentes(carpeta_seña):
    if not os.path.exists(carpeta_seña):
        return 0
    return len([f for f in os.listdir(carpeta_seña) if f.endswith(".npy")])


def mostrar_menu():
    print("\n" + "=" * 55)
    print("  MENU DE SEÑAS - escribe el numero y presiona Enter")
    print("=" * 55)
    for i, seña in enumerate(SEÑAS):
        carpeta_seña = os.path.join(CARPETA_DATASET, seña)
        n = contar_repeticiones_existentes(carpeta_seña)
        marca = "OK" if n >= REPETICIONES_OBJETIVO else "  "
        print(f"  [{marca}] {i+1:2d}. {seña:20s} ({n}/{REPETICIONES_OBJETIVO})")
    print("=" * 55)
    print("  Escribe un numero para grabar esa seña, o 'q' para salir")


def grabar_SEÑAS(indice_seña, cap, hand_detector, face_detector, pose_detector, timestamp_ms):
    seña_actual = SEÑAS[indice_seña]
    carpeta_seña = os.path.join(CARPETA_DATASET, seña_actual)
    os.makedirs(carpeta_seña, exist_ok=True)

    print(f"\nGrabando '{seña_actual}'. ESPACIO = grabar repeticion | ESC = volver al menu")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        timestamp_ms += 33
        r_manos, r_cara, r_pose = detectar_todo(frame, hand_detector, face_detector, pose_detector, timestamp_ms)
        hay_izq, hay_der, hay_cara, hay_pose = dibujar_deteccion_en_frame(frame, r_manos, r_cara, r_pose)

        num_reps = contar_repeticiones_existentes(carpeta_seña)
        cv2.putText(frame, f"Seña: {seña_actual} ({num_reps}/{REPETICIONES_OBJETIVO})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Cara:{'SI' if hay_cara else 'NO'} Pose:{'SI' if hay_pose else 'NO'} "
                            f"ManoI:{'SI' if hay_izq else 'NO'} ManoD:{'SI' if hay_der else 'NO'}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, "ESPACIO=grabar  ESC=volver al menu",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Grabacion de dataset ISN", frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27:  # ESC
            return timestamp_ms
        elif tecla == ord(" "):
            if not (hay_izq or hay_der):
                print("  Aviso: no se detecta ninguna mano en este momento, ajusta tu posicion.")

            for segundos in [2, 1]:
                ret, frame_cd = cap.read()
                frame_cd = cv2.flip(frame_cd, 1)
                timestamp_ms += 33
                r_m, r_c, r_p = detectar_todo(frame_cd, hand_detector, face_detector, pose_detector, timestamp_ms)
                dibujar_deteccion_en_frame(frame_cd, r_m, r_c, r_p)
                cv2.putText(frame_cd, f"Preparate... {segundos}", (130, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
                cv2.imshow("Grabacion de dataset ISN", frame_cd)
                cv2.waitKey(1)
                time.sleep(1)

            secuencia = []
            frames_sin_mano = 0
            for _ in range(FRAMES_POR_SECUENCIA):
                ret, frame_grab = cap.read()
                if not ret:
                    break
                frame_grab = cv2.flip(frame_grab, 1)
                timestamp_ms += 33
                r_m, r_c, r_p = detectar_todo(frame_grab, hand_detector, face_detector, pose_detector, timestamp_ms)
                izq, der, _, _ = dibujar_deteccion_en_frame(frame_grab, r_m, r_c, r_p)
                if not (izq or der):
                    frames_sin_mano += 1

                vector = landmarks_a_vector(r_m, r_c, r_p)
                secuencia.append(vector)

                cv2.putText(frame_grab, "GRABANDO...", (130, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                cv2.imshow("Grabacion de dataset ISN", frame_grab)
                cv2.waitKey(1)

            secuencia = np.array(secuencia)
            num_reps = contar_repeticiones_existentes(carpeta_seña)
            nombre_archivo = os.path.join(carpeta_seña, f"rep_{num_reps:03d}.npy")
            np.save(nombre_archivo, secuencia)

            aviso_calidad = ""
            if frames_sin_mano > FRAMES_POR_SECUENCIA * 0.3:
                aviso_calidad = f"  [AVISO: {frames_sin_mano}/{FRAMES_POR_SECUENCIA} frames sin ninguna mano detectada]"
            print(f"  Guardado: {nombre_archivo}  shape={secuencia.shape}{aviso_calidad}")


def main():
    descargar_modelos_si_faltan()
    hand_detector, face_detector, pose_detector = crear_detectores()
    os.makedirs(CARPETA_DATASET, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir la camara.")
        return

    timestamp_ms = 0

    while True:
        mostrar_menu()
        entrada = input("> ").strip().lower()

        if entrada == "q":
            break

        if not entrada.isdigit():
            print("Escribe un numero valido de la lista, o 'q' para salir.")
            continue

        indice = int(entrada) - 1
        if indice < 0 or indice >= len(SEÑAS):
            print("Numero fuera de rango.")
            continue

        timestamp_ms = grabar_SEÑAS(indice, cap, hand_detector, face_detector, pose_detector, timestamp_ms)

    cap.release()
    cv2.destroyAllWindows()
    print("Sesion de grabacion terminada.")


if __name__ == "__main__":
    main()