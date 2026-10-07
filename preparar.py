"""
PASO 2: Preparar los datos grabados para entrenamiento
------------------------------------------------------------
Este script:
  1. Lee dataset_trayectoria/<seña>/rep_XXX.npz
  2. Verifica formas y valores (descarta repeticiones defectuosas).
  3. Quita las señas que se quedaron sin repeticiones validas.
  4. Divide en ENTRENAMIENTO (70%), VALIDACION (15%) y PRUEBA (15%),
     estratificado por seña, con los mismos indices para secuencia y
     trayectoria.
  5. Calcula media y desviacion SOLO con entrenamiento y normaliza las tres
     particiones (columnas casi constantes no se escalan; los valores
     extremos se recortan).
  6. Guarda datos_preparados.npz y etiquetas.json.

El aumento de datos NO se hace aqui: se aplica al vuelo, solo sobre el
entrenamiento, en Entrenar.py (asi nunca contamina validacion ni
prueba).
"""

import os
import json

import numpy as np
from sklearn.model_selection import train_test_split

from caracteristicas import (
    FRAMES_POR_SECUENCIA, NUM_FEATURES_FRAME, DIM_TRAYECTORIA,
    calcular_estadisticas, normalizar,
)

CARPETA_DATASET = "dataset_trayectoria"
ARCHIVO_SALIDA = "datos_preparados.npz"
ARCHIVO_ETIQUETAS = "etiquetas.json"

PORCENTAJE_PRUEBA = 0.15
PORCENTAJE_VALIDACION = 0.15
MIN_REPETICIONES_POR_SENA = 10
SEMILLA = 42


def cargar_dataset():
    if not os.path.exists(CARPETA_DATASET):
        raise FileNotFoundError(
            f"No existe la carpeta '{CARPETA_DATASET}'. Graba primero con Grabacion.py"
        )

    senas = sorted(
        s for s in os.listdir(CARPETA_DATASET)
        if os.path.isdir(os.path.join(CARPETA_DATASET, s))
    )
    if not senas:
        raise ValueError(f"No hay carpetas de señas dentro de '{CARPETA_DATASET}'.")

    print(f"Señas encontradas: {senas}")

    secuencias, trayectorias, etiquetas = [], [], []
    descartadas = 0

    for sena in senas:
        carpeta = os.path.join(CARPETA_DATASET, sena)
        archivos = sorted(
            f for f in os.listdir(carpeta) if f.startswith("rep_") and f.endswith(".npz")
        )
        validas = 0

        for archivo in archivos:
            ruta = os.path.join(carpeta, archivo)
            try:
                datos = np.load(ruta)
                secuencia = datos["secuencia"]
                trayectoria = datos["trayectoria"]
            except Exception as e:
                print(f"    Descartada '{ruta}': no se pudo leer ({e})")
                descartadas += 1
                continue

            if secuencia.shape != (FRAMES_POR_SECUENCIA, NUM_FEATURES_FRAME):
                print(f"    Descartada '{ruta}': secuencia {secuencia.shape}, "
                      f"esperada {(FRAMES_POR_SECUENCIA, NUM_FEATURES_FRAME)}")
                descartadas += 1
                continue
            if trayectoria.shape != (DIM_TRAYECTORIA,):
                print(f"    Descartada '{ruta}': trayectoria {trayectoria.shape}, "
                      f"esperada {(DIM_TRAYECTORIA,)}")
                descartadas += 1
                continue
            if not (np.isfinite(secuencia).all() and np.isfinite(trayectoria).all()):
                print(f"    Descartada '{ruta}': contiene NaN o infinitos")
                descartadas += 1
                continue

            secuencias.append(secuencia)
            trayectorias.append(trayectoria)
            etiquetas.append(sena)
            validas += 1

        print(f"  {sena}: {validas} repeticiones validas")

    print(f"\nRepeticiones descartadas: {descartadas}")
    if not secuencias:
        raise ValueError("No quedaron repeticiones validas.")
    return secuencias, trayectorias, etiquetas


def main():
    secuencias, trayectorias, etiquetas_texto = cargar_dataset()

    # Solo señas con al menos una repeticion valida (evita clases vacias)
    lista_senas = sorted(set(etiquetas_texto))
    sena_a_indice = {sena: i for i, sena in enumerate(lista_senas)}

    X = np.array(secuencias, dtype=np.float32)
    T = np.array(trayectorias, dtype=np.float32)
    y = np.array([sena_a_indice[e] for e in etiquetas_texto])

    conteos = np.bincount(y, minlength=len(lista_senas))
    pocas = [f"{lista_senas[i]} ({c})" for i, c in enumerate(conteos) if c < MIN_REPETICIONES_POR_SENA]
    if pocas:
        raise ValueError(
            f"Estas señas tienen menos de {MIN_REPETICIONES_POR_SENA} repeticiones validas: "
            f"{', '.join(pocas)}. Graba mas antes de continuar."
        )

    print(f"\nX: {X.shape}   T: {T.shape}   y: {y.shape}")
    print(f"Señas con datos: {len(lista_senas)}")

    # --- Split estratificado: prueba, luego validacion sobre lo que queda ---
    indices = np.arange(len(y))
    try:
        idx_trainval, idx_test = train_test_split(
            indices, test_size=PORCENTAJE_PRUEBA, random_state=SEMILLA, stratify=y
        )
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=PORCENTAJE_VALIDACION / (1.0 - PORCENTAJE_PRUEBA),
            random_state=SEMILLA,
            stratify=y[idx_trainval],
        )
    except ValueError as e:
        raise ValueError(
            f"No se pudo dividir el dataset de forma estratificada: {e}\n"
            f"Normalmente significa que hay muy pocas repeticiones para tantas señas."
        )

    print(f"Entrenamiento: {len(idx_train)} | Validacion: {len(idx_val)} | Prueba: {len(idx_test)}")

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    T_train, T_val, T_test = T[idx_train], T[idx_val], T[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    # --- Normalizacion con estadisticas de ENTRENAMIENTO ---
    media_X, desviacion_X = calcular_estadisticas(X_train, ejes=(0, 1))
    media_T, desviacion_T = calcular_estadisticas(T_train, ejes=0)

    X_train = normalizar(X_train, media_X, desviacion_X)
    X_val = normalizar(X_val, media_X, desviacion_X)
    X_test = normalizar(X_test, media_X, desviacion_X)
    T_train = normalizar(T_train, media_T, desviacion_T)
    T_val = normalizar(T_val, media_T, desviacion_T)
    T_test = normalizar(T_test, media_T, desviacion_T)

    np.savez_compressed(
        ARCHIVO_SALIDA,
        X_train=X_train, X_val=X_val, X_test=X_test,
        T_train=T_train, T_val=T_val, T_test=T_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        media_X=media_X, desviacion_X=desviacion_X,
        media_T=media_T, desviacion_T=desviacion_T,
    )

    with open(ARCHIVO_ETIQUETAS, "w", encoding="utf-8") as f:
        json.dump(sena_a_indice, f, ensure_ascii=False, indent=2)

    print(f"\nListo. Datos guardados en '{ARCHIVO_SALIDA}' y etiquetas en '{ARCHIVO_ETIQUETAS}'")


if __name__ == "__main__":
    main()