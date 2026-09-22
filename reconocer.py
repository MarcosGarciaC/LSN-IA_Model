"""
PASO 4: Reconocimiento en tiempo real + salida de audio
------------------------------------------------------------
Este es el ultimo bloque: conecta todo lo que ya construiste.

Como funciona:
  1. Captura landmarks en vivo (igual que en los pasos anteriores)
  2. Acumula los ultimos 30 frames en una "ventana deslizante"
     (una especie de memoria corta de lo que acaba de pasar)
  3. Cada cierto numero de frames, le pasa esa ventana al modelo
     entrenado y le pregunta que sena cree que es
  4. Si el modelo esta razonablemente seguro (por encima de un umbral
     de confianza), muestra el texto en pantalla Y lo dice en voz alta
  5. Evita repetir el mismo audio en bucle mientras sigues haciendo
     la misma sena

Controles:
  - Presiona 'q' para salir.

CAMBIO: ahora usa RunningMode.VIDEO en vez de IMAGE (igual que
grabar_dataset.py), para que el detector tenga memoria entre frames.
Es importante que ambos scripts usen el MISMO modo de deteccion,
para que lo que el modelo aprendio se parezca a lo que ve en vivo.
"""

import os
import json
import time
import urllib.request
import threading
import queue

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pyttsx3

ARCHIVO_MODELO = "modelo_señas.pt"
FRAMES_POR_SECUENCIA = 30
UMBRAL_CONFIANZA = 0.75       # solo acepta predicciones con esta seguridad o mas
SEGUNDOS_ENTRE_PREDICCIONES = 1.0  # no prediga en cada frame, para no saturar
SEGUNDOS_ANTES_DE_REPETIR = 3.0    # evita decir la misma sena en bucle

# Etiquetas que el modelo puede reconocer pero que NUNCA deben decirse en voz
# alta (ej: la clase de "reposo" que representa "no se esta haciendo ninguna
# sena"). Se muestran en pantalla igual, para que puedas ver que el modelo
# las esta detectando correctamente, pero se omiten del audio.
ETIQUETAS_SILENCIOSAS = {"reposo", "nada"}

MODELOS_MEDIAPIPE = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

N_LANDMARKS_MANO = 21
N_LANDMARKS_CARA = 478
N_LANDMARKS_POSE = 33


# --- Misma arquitectura de modelo usada en el entrenamiento  ---
class ModeloLSTM(nn.Module):
    def __init__(self, num_features, num_clases, tamano_oculto=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=tamano_oculto,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        self.clasificador = nn.Sequential(
            nn.Linear(tamano_oculto, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_clases),
        )

    def forward(self, x):
        salida_lstm, (h_n, c_n) = self.lstm(x)
        ultimo_estado = h_n[-1]
        return self.clasificador(ultimo_estado)


# --- Motor de texto a voz corriendo en un hilo aparte, para no congelar el video ---
audio = queue.Queue()

def hilo_audio():
    # Nota: se crea un motor NUEVO cada vez que se habla, en vez de reutilizar
    # uno solo. Esto evita un bug conocido de pyttsx3 en Windows donde el
    # motor deja de responder despues de la primera llamada a runAndWait()
    # si se reutiliza el mismo objeto repetidamente.
    while True:
        texto = audio.get()
        if texto is None:
            break
        motor = pyttsx3.init()
        motor.setProperty("rate", 165)
        motor.say(texto)
        motor.runAndWait()
        motor.stop()
        del motor

def decir(texto):
    audio.put(texto)


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
    return (
        hand_detector.detect_for_video(mp_image, timestamp_ms),
        face_detector.detect_for_video(mp_image, timestamp_ms),
        pose_detector.detect_for_video(mp_image, timestamp_ms),
    )


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


def main():
    # --- Cargar el modelo entrenado ---
    if not os.path.exists(ARCHIVO_MODELO):
        print(f"ERROR: no se encontro '{ARCHIVO_MODELO}'. Corre primero entrenar_modelo.py")
        return

    checkpoint = torch.load(ARCHIVO_MODELO, map_location="cpu")
    num_features = checkpoint["num_features"]
    num_clases = checkpoint["num_clases"]
    sena_a_indice = checkpoint["sena_a_indice"]
    indice_a_sena = {v: k for k, v in sena_a_indice.items()}

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = ModeloLSTM(num_features=num_features, num_clases=num_clases).to(dispositivo)
    modelo.load_state_dict(checkpoint["modelo_state_dict"])
    modelo.eval()

    print(f"Modelo cargado. Senas conocidas: {list(sena_a_indice.keys())}")

    # --- Iniciar hilo de audio ---
    hilo = threading.Thread(target=hilo_audio, daemon=True)
    hilo.start()

    # --- Preparar deteccion ---
    descargar_modelos_si_faltan()
    hand_detector, face_detector, pose_detector = crear_detectores()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir la camara.")
        return

    ventana_frames = []
    ultima_prediccion_tiempo = 0
    ultima_sena_dicha = None
    ultima_sena_tiempo = 0
    texto_en_pantalla = "..."
    timestamp_ms = 0  # contador creciente requerido por RunningMode.VIDEO

    print("Reconocimiento iniciado. Presiona 'q' para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        timestamp_ms += 33
        r_manos, r_cara, r_pose = detectar_todo(frame, hand_detector, face_detector, pose_detector, timestamp_ms)

        # Dibujar landmarks para ver que se esta detectando
        for mano_landmarks in r_manos.hand_landmarks:
            dibujar_puntos(frame, mano_landmarks, (0, 255, 0), radio=3)
        for cara_landmarks in r_cara.face_landmarks:
            dibujar_puntos(frame, cara_landmarks, (255, 255, 0), radio=1)
        for pose_landmarks in r_pose.pose_landmarks:
            dibujar_puntos(frame, pose_landmarks, (0, 0, 255), radio=3)

        vector = landmarks_a_vector(r_manos, r_cara, r_pose)
        ventana_frames.append(vector)
        if len(ventana_frames) > FRAMES_POR_SECUENCIA:
            ventana_frames.pop(0)  # mantener solo los ultimos 30 frames

        ahora = time.time()

        # Solo predecir si ya tenemos suficientes frames Y ya paso el intervalo minimo
        if len(ventana_frames) == FRAMES_POR_SECUENCIA and \
           (ahora - ultima_prediccion_tiempo) >= SEGUNDOS_ENTRE_PREDICCIONES:

            ultima_prediccion_tiempo = ahora

            entrada = torch.tensor(np.array([ventana_frames]), dtype=torch.float32).to(dispositivo)
            with torch.no_grad():
                salida = modelo(entrada)
                probabilidades = torch.softmax(salida, dim=1)
                confianza, indice_predicho = torch.max(probabilidades, dim=1)
                confianza = confianza.item()
                indice_predicho = indice_predicho.item()

            if confianza >= UMBRAL_CONFIANZA:
                sena_predicha = indice_a_sena[indice_predicho]

                if sena_predicha in ETIQUETAS_SILENCIOSAS:
                    # Se reconoce y se muestra en pantalla, pero nunca se dice
                    # en voz alta (ej: "reposo" = no se esta haciendo ninguna sena)
                    texto_en_pantalla = f"{sena_predicha} ({confianza*100:.0f}%) [silenciado]"
                    ultima_sena_dicha = None  # permite que la siguiente sena real se diga de inmediato
                else:
                    texto_en_pantalla = f"{sena_predicha} ({confianza*100:.0f}%)"

                    # Evitar repetir la misma sena en bucle muy seguido
                    es_sena_distinta = sena_predicha != ultima_sena_dicha
                    paso_tiempo_suficiente = (ahora - ultima_sena_tiempo) >= SEGUNDOS_ANTES_DE_REPETIR

                    if es_sena_distinta or paso_tiempo_suficiente:
                        decir(sena_predicha)
                        ultima_sena_dicha = sena_predicha
                        ultima_sena_tiempo = ahora
            else:
                texto_en_pantalla = f"(sin reconocer, {confianza*100:.0f}%)"

        # --- Mostrar resultado en pantalla ---
        cv2.rectangle(frame, (0, 0), (640, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"Prediccion: {texto_en_pantalla}", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Traductor ISN - tiempo real", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    audio.put(None)  # senal para cerrar el hilo de audio
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()