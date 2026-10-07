"""
PASO 4: Reconocimiento en tiempo real + salida de audio
------------------------------------------------------------
Como funciona:
  1. Captura la camara y calcula la referencia corporal (cabeza, hombros,
     cintura) con el MISMO codigo de grabacion (caracteristicas.py).
  2. Toma muestras a FPS_SECUENCIA por segundo y mantiene una ventana
     deslizante con las ultimas 30 (2 segundos).
  3. Cada SEGUNDOS_ENTRE_PREDICCIONES construye la secuencia y la
     trayectoria de esa ventana, las normaliza con las estadisticas del
     entrenamiento y le pregunta al modelo.
  4. Una seña se acepta solo si la confianza supera el umbral Y se repite
     en VOTOS_CONSECUTIVOS predicciones seguidas (evita falsos positivos).
  5. Se muestra en pantalla y se dice en voz alta (sin repetir en bucle).

Controles: 'q' o ESC para salir.
Archivos necesarios en la misma carpeta: caracteristicas.py, modelo.py,
modelo_señas.pt y los .task de MediaPipe (se descargan solos).
"""

import os
import time
import threading
import queue
from collections import deque

import cv2
import numpy as np
import torch

from caracteristicas import (
    FRAMES_POR_SECUENCIA, FPS_SECUENCIA, NUM_FEATURES_FRAME, DIM_TRAYECTORIA,
    Reloj, Muestreador,
    descargar_modelos_si_faltan, crear_detectores,
    leer_y_detectar, construir_secuencia, normalizar,
    dibujar_overlay,
)
from modelo import ModeloDosRamas

try:
    import pyttsx3
    AUDIO_DISPONIBLE = True
except ImportError:
    AUDIO_DISPONIBLE = False

ARCHIVO_MODELO = "modelo_señas.pt"

UMBRAL_CONFIANZA = 0.75
SEGUNDOS_ENTRE_PREDICCIONES = 0.5
VOTOS_CONSECUTIVOS = 2             # predicciones iguales seguidas para aceptar
SEGUNDOS_ANTES_DE_REPETIR = 3.0
MAX_FRACCION_SIN_MANOS = 0.9       # si casi toda la ventana no tiene manos, no se predice

# Se muestran en pantalla pero nunca se dicen en voz alta
ETIQUETAS_SILENCIOSAS = {"reposo", "nada"}

VENTANA = "Traductor ISN - tiempo real"


# --- Audio en un hilo aparte para no congelar el video ---
audio = queue.Queue()


def hilo_audio():
    # Se crea un motor NUEVO por cada frase: evita un bug de pyttsx3 en Windows
    # donde el motor deja de responder tras la primera llamada a runAndWait().
    while True:
        texto = audio.get()
        if texto is None:
            break
        try:
            motor = pyttsx3.init()
            motor.setProperty("rate", 165)
            motor.say(texto)
            motor.runAndWait()
            motor.stop()
            del motor
        except Exception as e:
            print(f"(error de audio: {e})")


def decir(texto):
    if AUDIO_DISPONIBLE:
        audio.put(texto.replace("_", " "))


def cargar_modelo(dispositivo):
    checkpoint = torch.load(ARCHIVO_MODELO, map_location="cpu")

    if checkpoint["frames_por_secuencia"] != FRAMES_POR_SECUENCIA or \
       checkpoint.get("fps_secuencia") != FPS_SECUENCIA:
        raise ValueError(
            "El modelo se entreno con otro muestreo "
            f"({checkpoint['frames_por_secuencia']} muestras a {checkpoint.get('fps_secuencia')} fps) "
            f"que el configurado en caracteristicas.py ({FRAMES_POR_SECUENCIA} a {FPS_SECUENCIA} fps)."
        )
    if checkpoint["num_features"] != NUM_FEATURES_FRAME or checkpoint["dim_trayectoria"] != DIM_TRAYECTORIA:
        raise ValueError("Las dimensiones del modelo no coinciden con caracteristicas.py.")

    modelo = ModeloDosRamas(
        num_features=checkpoint["num_features"],
        dim_trayectoria=checkpoint["dim_trayectoria"],
        num_clases=checkpoint["num_clases"],
        tamano_oculto=checkpoint["tamano_oculto"],
        num_capas=checkpoint["num_capas_lstm"],
        dropout=checkpoint["dropout"],
        tamano_trayectoria_oculto=checkpoint["tamano_trayectoria_oculto"],
    ).to(dispositivo)
    modelo.load_state_dict(checkpoint["modelo_state_dict"])
    modelo.eval()

    estadisticas = {
        "media_X": checkpoint["media_X"].numpy(),
        "desviacion_X": checkpoint["desviacion_X"].numpy(),
        "media_T": checkpoint["media_T"].numpy(),
        "desviacion_T": checkpoint["desviacion_T"].numpy(),
    }
    indice_a_sena = {v: k for k, v in checkpoint["sena_a_indice"].items()}
    return modelo, estadisticas, indice_a_sena


def predecir(modelo, estadisticas, indice_a_sena, ventana, dispositivo):
    """Devuelve ('sin_manos', None) | ('sin_ref', None) | ('ok', lista top-3)."""
    resultado = construir_secuencia(list(ventana))
    if resultado is None:
        return "sin_ref", None
    secuencia, trayectoria, _, info = resultado

    if info["frames_sin_mano"] >= FRAMES_POR_SECUENCIA * MAX_FRACCION_SIN_MANOS:
        return "sin_manos", None

    X = normalizar(secuencia, estadisticas["media_X"], estadisticas["desviacion_X"])
    T = normalizar(trayectoria, estadisticas["media_T"], estadisticas["desviacion_T"])

    x = torch.tensor(X[None], dtype=torch.float32).to(dispositivo)
    t = torch.tensor(T[None], dtype=torch.float32).to(dispositivo)
    with torch.no_grad():
        probabilidades = torch.softmax(modelo(x, t), dim=1)[0].cpu().numpy()

    orden = np.argsort(probabilidades)[::-1][:3]
    top = [(indice_a_sena[int(i)], float(probabilidades[i])) for i in orden]
    return "ok", top


def main():
    if not os.path.exists(ARCHIVO_MODELO):
        print(f"ERROR: no se encontro '{ARCHIVO_MODELO}'. Corre primero Entrenar.py")
        return

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        modelo, estadisticas, indice_a_sena = cargar_modelo(dispositivo)
    except ValueError as e:
        print(f"ERROR: {e}")
        return
    print(f"Modelo cargado. Señas conocidas: {list(indice_a_sena.values())}")

    if AUDIO_DISPONIBLE:
        threading.Thread(target=hilo_audio, daemon=True).start()
    else:
        print("(pyttsx3 no esta instalado: se mostrara el texto pero sin audio. "
              "Instala con: pip install pyttsx3)")

    descargar_modelos_si_faltan()
    detectores = crear_detectores()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir la camara.")
        return

    reloj = Reloj()
    muestreador = Muestreador(FPS_SECUENCIA)
    ventana = deque(maxlen=FRAMES_POR_SECUENCIA)
    historial = deque(maxlen=VOTOS_CONSECUTIVOS)
    ref = None

    ultima_prediccion = 0.0
    ultima_sena_dicha = None
    ultima_sena_tiempo = 0.0
    texto = "llenando ventana..."
    top3 = []

    print("Reconocimiento iniciado. Presiona 'q' o ESC para salir.")

    while True:
        res = leer_y_detectar(cap, detectores, reloj, ref)
        if res is None:
            break
        frame, manos, ref = res
        dibujar_overlay(frame, manos, ref)

        if muestreador.toca():
            ventana.append({"manos": manos, "ref": ref})

        ahora = time.monotonic()
        if len(ventana) < FRAMES_POR_SECUENCIA:
            texto = f"llenando ventana... {len(ventana)}/{FRAMES_POR_SECUENCIA}"
        elif ahora - ultima_prediccion >= SEGUNDOS_ENTRE_PREDICCIONES:
            ultima_prediccion = ahora
            estado, top3_nuevo = predecir(modelo, estadisticas, indice_a_sena, ventana, dispositivo)

            if estado != "ok":
                texto = "(sin manos)" if estado == "sin_manos" else "(sin referencia corporal)"
                top3 = []
                historial.clear()
                ultima_sena_dicha = None   # la siguiente seña real se dice de inmediato
            else:
                top3 = top3_nuevo
                sena, confianza = top3[0]

                if confianza < UMBRAL_CONFIANZA:
                    texto = f"(sin reconocer, {confianza*100:.0f}%)"
                    historial.append(None)
                else:
                    historial.append(sena)
                    aceptada = (len(historial) == VOTOS_CONSECUTIVOS
                                and all(h == sena for h in historial))

                    if sena in ETIQUETAS_SILENCIOSAS:
                        texto = f"{sena} ({confianza*100:.0f}%) [silenciado]"
                        if aceptada:
                            ultima_sena_dicha = None
                    else:
                        texto = f"{sena} ({confianza*100:.0f}%)" + ("" if aceptada else " ...")
                        if aceptada:
                            es_distinta = sena != ultima_sena_dicha
                            paso_tiempo = (ahora - ultima_sena_tiempo) >= SEGUNDOS_ANTES_DE_REPETIR
                            if es_distinta or paso_tiempo:
                                decir(sena)
                                ultima_sena_dicha = sena
                                ultima_sena_tiempo = ahora

        # --- Pantalla ---
        ancho = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (ancho, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"Prediccion: {texto}", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for i, (nombre, prob) in enumerate(top3):
            cv2.putText(frame, f"{i+1}. {nombre} {prob*100:.0f}%",
                        (10, frame.shape[0] - 15 - 22 * (len(top3) - 1 - i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(VENTANA, frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break

    audio.put(None)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()