"""
PASO 1: Grabacion de dataset basada en TRAYECTORIA relativa al cuerpo
--------------------------------------------------------------------------
Flujo:
  1. Menu en terminal con las señas y cuantas repeticiones llevas.
  2. Escribes el numero de la seña y Enter.
  3. Se abre la camara con las lineas de referencia dibujadas en vivo
     (cabeza, hombros, cintura y eje vertical; punteadas = estimadas).
  4. ESPACIO = grabar una repeticion (cuenta regresiva + 30 muestras).
     B       = borrar la ultima repeticion de esa seña.
     ESC     = volver al menu.
  5. 'q' en el menu para salir.

Cada repeticion dura 2 segundos (30 muestras a 15 por segundo): haz la seña
dentro de ese tiempo. Se guarda como rep_XXX.npz con:
   secuencia   (30, 157)  -> para la LSTM
   trayectoria (178,)     -> recorrido completo de ambas muñecas + resumen
   zonas       (30, 4)    -> zona vertical/horizontal de cada muñeca

Todo el calculo de caracteristicas esta en caracteristicas.py (compartido
con reconocer.py). Mantén ambos archivos en la misma carpeta.

  comando para activar el entorno: venv\\Scripts\\activate
"""

import os
import time

import cv2
import numpy as np

from caracteristicas import (
    FRAMES_POR_SECUENCIA, FPS_SECUENCIA,
    Reloj, Muestreador,
    descargar_modelos_si_faltan, crear_detectores,
    leer_y_detectar, construir_secuencia,
    dibujar_overlay, describir_zonas,
)

# ============================================================
# CONFIGURACION - agrega aqui todas tus SEÑAS
# ============================================================
SEÑAS = [
    "hola", "gracias", "por_favor", "adios", "buenos_dias",
    "J", "nada", "reposo",
    "cuerpo",
    "hombre",
    "mujer",
    "cabeza",
    "cabello",
    "ojos",
    "nariz",
    "oreja",
    "boca",
    "pecho",
    "brazo",
    "mano",
    "estomago",
    "pierna",
    "pies",
    "amigo",
]

CARPETA_DATASET = "dataset_trayectoria"
REPETICIONES_OBJETIVO = 25
VENTANA = "Grabacion de dataset ISN"


def archivos_repeticion(carpeta):
    if not os.path.exists(carpeta):
        return []
    return sorted(f for f in os.listdir(carpeta) if f.startswith("rep_") and f.endswith(".npz"))


def contar_repeticiones_existentes(carpeta_seña):
    return len(archivos_repeticion(carpeta_seña))


def siguiente_nombre(carpeta_seña):
    """Siguiente rep_XXX libre (usa el numero mas alto + 1, no el conteo)."""
    numeros = []
    for f in archivos_repeticion(carpeta_seña):
        try:
            numeros.append(int(f[4:-4]))
        except ValueError:
            pass
    n = (max(numeros) + 1) if numeros else 0
    return os.path.join(carpeta_seña, f"rep_{n:03d}.npz")


def mostrar_menu():
    print("\n" + "=" * 55)
    print("  MENU DE SEÑAS - escribe el numero y presiona Enter")
    print("=" * 55)
    for i, seña in enumerate(SEÑAS):
        n = contar_repeticiones_existentes(os.path.join(CARPETA_DATASET, seña))
        marca = "OK" if n >= REPETICIONES_OBJETIVO else "  "
        print(f"  [{marca}] {i+1:2d}. {seña:20s} ({n}/{REPETICIONES_OBJETIVO})")
    print("=" * 55)
    print("  Escribe un numero para grabar esa seña, o 'q' para salir")


def grabar_repeticion(cap, detectores, reloj, ref, carpeta_seña):
    # --- Cuenta regresiva (la deteccion sigue corriendo para mantener el tracking) ---
    for segundos in (2, 1):
        fin = time.monotonic() + 1.0
        while time.monotonic() < fin:
            res = leer_y_detectar(cap, detectores, reloj, ref)
            if res is None:
                return ref
            frame, manos, ref = res
            dibujar_overlay(frame, manos, ref)
            cv2.putText(frame, f"Preparate... {segundos}", (130, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
            cv2.imshow(VENTANA, frame)
            cv2.waitKey(1)

    # --- Grabacion: 30 muestras a FPS_SECUENCIA por segundo de reloj real ---
    raws = []
    estelas = {"izq": [], "der": []}
    muestreador = Muestreador(FPS_SECUENCIA)
    t_inicio = time.monotonic()
    while len(raws) < FRAMES_POR_SECUENCIA:
        res = leer_y_detectar(cap, detectores, reloj, ref)
        if res is None:
            break
        frame, manos, ref = res

        if muestreador.toca():
            raws.append({"manos": manos, "ref": ref})
            for nombre in ("izq", "der"):
                if manos[nombre] is not None:
                    estelas[nombre].append((int(manos[nombre][0][0]), int(manos[nombre][0][1])))

        dibujar_overlay(frame, manos, ref, estelas)
        cv2.putText(frame, f"GRABANDO... {len(raws)}/{FRAMES_POR_SECUENCIA}", (130, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        cv2.imshow(VENTANA, frame)
        cv2.waitKey(1)
    duracion = time.monotonic() - t_inicio

    if len(raws) < FRAMES_POR_SECUENCIA:
        print(f"  DESCARTADA: solo se capturaron {len(raws)}/{FRAMES_POR_SECUENCIA} muestras.")
        return ref

    resultado = construir_secuencia(raws)
    if resultado is None:
        print("  DESCARTADA: no se detecto ninguna referencia corporal (cara u hombros).")
        return ref
    secuencia, trayectoria, zonas, info = resultado

    ruta = siguiente_nombre(carpeta_seña)
    np.savez_compressed(ruta, secuencia=secuencia, trayectoria=trayectoria, zonas=zonas)

    avisos = []
    if info["frames_sin_mano"] > FRAMES_POR_SECUENCIA * 0.3:
        avisos.append(f"{info['frames_sin_mano']}/{FRAMES_POR_SECUENCIA} muestras sin ninguna mano")
    fps_real = FRAMES_POR_SECUENCIA / max(duracion, 1e-6)
    if fps_real < FPS_SECUENCIA * 0.8:
        avisos.append(f"tu PC solo logro {fps_real:.1f} muestras/s (objetivo {FPS_SECUENCIA}); "
                      f"baja FPS_SECUENCIA en caracteristicas.py")
    aviso = f"  [AVISO: {'; '.join(avisos)}]" if avisos else ""
    print(f"  Guardado: {ruta}  duracion={duracion:.1f}s  "
          f"hombros reales={info['frac_hombros_reales']*100:.0f}%  "
          f"cintura real={info['frac_cintura_real']*100:.0f}%{aviso}")
    return ref


def grabar_seña(indice_seña, cap, detectores, reloj):
    seña_actual = SEÑAS[indice_seña]
    carpeta_seña = os.path.join(CARPETA_DATASET, seña_actual)
    os.makedirs(carpeta_seña, exist_ok=True)

    print(f"\nGrabando '{seña_actual}'. ESPACIO = grabar | B = borrar ultima | ESC = menu")
    ref = None

    while True:
        res = leer_y_detectar(cap, detectores, reloj, ref)
        if res is None:
            break
        frame, manos, ref = res
        dibujar_overlay(frame, manos, ref)

        num_reps = contar_repeticiones_existentes(carpeta_seña)
        cv2.putText(frame, f"Sena: {seña_actual} ({num_reps}/{REPETICIONES_OBJETIVO})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        estado_ref = "sin referencia"
        if ref is not None:
            estado_ref = (f"hombros {'REAL' if ref['hombros_real'] else 'EST'}  "
                          f"cintura {'REAL' if ref['cintura_real'] else 'EST'}")
        cv2.putText(frame, estado_ref, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, describir_zonas(manos, ref), (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "ESPACIO=grabar  B=borrar ultima  ESC=menu",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(VENTANA, frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27:  # ESC
            return
        elif tecla == ord("b"):
            archivos = archivos_repeticion(carpeta_seña)
            if archivos:
                os.remove(os.path.join(carpeta_seña, archivos[-1]))
                print(f"  Borrada: {archivos[-1]}")
        elif tecla == ord(" "):
            if manos["izq"] is None and manos["der"] is None:
                print("  Aviso: no se detecta ninguna mano en este momento, ajusta tu posicion.")
            ref = grabar_repeticion(cap, detectores, reloj, ref, carpeta_seña)


def main():
    descargar_modelos_si_faltan()
    detectores = crear_detectores()
    os.makedirs(CARPETA_DATASET, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir la camara.")
        return

    reloj = Reloj()

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

        grabar_seña(indice, cap, detectores, reloj)

    cap.release()
    cv2.destroyAllWindows()
    print("Sesion de grabacion terminada.")


if __name__ == "__main__":
    main()