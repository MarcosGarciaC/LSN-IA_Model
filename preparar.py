"""
PASO 2: Preparar los datos grabados para entrenamiento
------------------------------------------------------------

Este script:
  1. Lee todas las carpetas dentro de dataset_trayectoria/
     (cada carpeta = una seña).
  2. Carga los archivos rep_XXX.npz.
  3. De cada archivo obtiene:
       - secuencia: matriz (30, 157)
       - trayectoria: vector 1D
  4. Verifica que las formas sean consistentes.
  5. Descarta archivos con formas incorrectas.
  6. Convierte los nombres de señas a números.
  7. Divide los índices en entrenamiento (80%) y prueba (20%).
  8. Usa los mismos índices para secuencias y trayectorias.
  9. Calcula media y desviación estándar usando SOLO entrenamiento.
 10. Normaliza entrenamiento y prueba usando esas estadísticas.
 11. Guarda todos los datos y estadísticas en datos_preparados.npz.
 12. Guarda el mapeo de etiquetas en etiquetas.json.

Opcionalmente:
  - Se puede activar aumento de datos SOLO para entrenamiento.
"""

import os
import json

import numpy as np
from sklearn.model_selection import train_test_split


CARPETA_DATASET = "dataset_trayectoria"
ARCHIVO_SALIDA = "datos_preparados.npz"
ARCHIVO_ETIQUETAS = "etiquetas.json"

# Aumento de datos.
# False = desactivado
# True = activado
USAR_AUMENTO_DATOS = False


def cargar_dataset():
    """
    Recorre dataset_trayectoria/<seña>/rep_XXX.npz.

    De cada archivo obtiene:
        - secuencia (30, 157)
        - trayectoria (vector 1D)

    Devuelve listas paralelas de secuencias, trayectorias y etiquetas.
    """

    if not os.path.exists(CARPETA_DATASET):
        raise FileNotFoundError(
            f"No existe la carpeta '{CARPETA_DATASET}'. "
            f"Verifica que el dataset haya sido generado."
        )

    senas = sorted(os.listdir(CARPETA_DATASET))

    senas = [
        s for s in senas
        if os.path.isdir(os.path.join(CARPETA_DATASET, s))
    ]

    if len(senas) == 0:
        raise ValueError(
            "No hay ninguna carpeta de señas dentro de "
            "dataset_trayectoria/."
        )

    print(f"Señas encontradas: {senas}")

    secuencias = []
    trayectorias = []
    etiquetas_texto = []

    for seña in senas:

        carpeta_sena = os.path.join(CARPETA_DATASET, seña)

        archivos = sorted([
            f
            for f in os.listdir(carpeta_sena)
            if f.startswith("rep_") and f.endswith(".npz")
        ])

        if len(archivos) == 0:
            print(
                f"  Aviso: '{seña}' no tiene repeticiones válidas. "
                f"Se omite."
            )
            continue

        print(f"  {seña}: {len(archivos)} repeticiones")

        for archivo in archivos:

            ruta = os.path.join(carpeta_sena, archivo)

            try:
                datos = np.load(ruta)

                if "secuencia" not in datos:
                    print(
                        f"    Aviso: '{archivo}' no contiene "
                        f"'secuencia'. Se descarta."
                    )
                    continue

                if "trayectoria" not in datos:
                    print(
                        f"    Aviso: '{archivo}' no contiene "
                        f"'trayectoria'. Se descarta."
                    )
                    continue

                secuencia = datos["secuencia"]
                trayectoria = datos["trayectoria"]

                secuencias.append(secuencia)
                trayectorias.append(trayectoria)
                etiquetas_texto.append(seña)

            except Exception as e:
                print(
                    f"    Error leyendo '{ruta}': {e}. "
                    f"Se descarta."
                )

    if len(secuencias) == 0:
        raise ValueError(
            "No se encontraron archivos .npz válidos."
        )

    return secuencias, trayectorias, etiquetas_texto, senas


def verificar_formas(
    secuencias,
    trayectorias,
    etiquetas_texto
):
    """
    Verifica que:
      - cada secuencia tenga exactamente 30 frames
      - cada secuencia tenga 157 características
      - todas las trayectorias tengan la misma longitud

    Si una repetición no cumple, se descarta completa.
    """

    if len(secuencias) != len(trayectorias):
        raise ValueError(
            "El número de secuencias y trayectorias no coincide."
        )

    longitudes_trayectoria = [
        t.shape[0]
        for t in trayectorias
    ]

    if len(longitudes_trayectoria) == 0:
        raise ValueError(
            "No hay trayectorias para verificar."
        )

    longitud_trayectoria_esperada = max(
        set(longitudes_trayectoria),
        key=longitudes_trayectoria.count
    )

    secuencias_validas = []
    trayectorias_validas = []
    etiquetas_validas = []

    descartadas = 0

    for secuencia, trayectoria, etiqueta in zip(
        secuencias,
        trayectorias,
        etiquetas_texto
    ):

        forma_secuencia_correcta = (
            secuencia.ndim == 2
            and secuencia.shape == (30, 157)
        )

        forma_trayectoria_correcta = (
            trayectoria.ndim == 1
            and trayectoria.shape[0]
            == longitud_trayectoria_esperada
        )

        if not forma_secuencia_correcta:
            print(
                f"  Aviso: repetición de '{etiqueta}' descartada. "
                f"Forma de secuencia incorrecta: "
                f"{secuencia.shape}. Esperada: (30, 157)"
            )

            descartadas += 1
            continue

        if not forma_trayectoria_correcta:
            print(
                f"  Aviso: repetición de '{etiqueta}' descartada. "
                f"Longitud de trayectoria incorrecta: "
                f"{trayectoria.shape}. "
                f"Esperada: ({longitud_trayectoria_esperada},)"
            )

            descartadas += 1
            continue

        secuencias_validas.append(secuencia)
        trayectorias_validas.append(trayectoria)
        etiquetas_validas.append(etiqueta)

    print(
        f"\nRepeticiones descartadas por formas incorrectas: "
        f"{descartadas}"
    )

    print(
        f"Repeticiones válidas: {len(secuencias_validas)}"
    )

    return (
        secuencias_validas,
        trayectorias_validas,
        etiquetas_validas,
        longitud_trayectoria_esperada
    )


def normalizar_datos(
    X_train,
    X_test,
    T_train,
    T_test
):
    """
    Calcula estadísticas SOLO con entrenamiento.

    Para secuencias:
        media y desviación por característica.
        Se consideran todos los frames del entrenamiento.

    Para trayectorias:
        media y desviación por característica.

    Se agrega epsilon para evitar división entre cero.
    """

    epsilon = 1e-8

    # ---------------------------------------------------------
    # SECUENCIAS
    # ---------------------------------------------------------

    # X_train:
    # (ejemplos, 30 frames, 157 características)

    media_X = np.mean(
        X_train,
        axis=(0, 1)
    )

    desviacion_X = np.std(
        X_train,
        axis=(0, 1)
    )

    desviacion_X = desviacion_X + epsilon

    X_train_normalizado = (
        X_train - media_X
    ) / desviacion_X

    X_test_normalizado = (
        X_test - media_X
    ) / desviacion_X

    # ---------------------------------------------------------
    # TRAYECTORIAS
    # ---------------------------------------------------------

    # T_train:
    # (ejemplos, características_trayectoria)

    media_T = np.mean(
        T_train,
        axis=0
    )

    desviacion_T = np.std(
        T_train,
        axis=0
    )

    desviacion_T = desviacion_T + epsilon

    T_train_normalizado = (
        T_train - media_T
    ) / desviacion_T

    T_test_normalizado = (
        T_test - media_T
    ) / desviacion_T

    return (
        X_train_normalizado,
        X_test_normalizado,
        T_train_normalizado,
        T_test_normalizado,
        media_X,
        desviacion_X,
        media_T,
        desviacion_T
    )


def aumentar_datos(
    X_train,
    T_train,
    y_train
):
    """
    Aumento de datos opcional.

    Se aplica exclusivamente al conjunto de entrenamiento.

    Genera pequeñas variaciones:
      - ruido en las posiciones
      - pequeños cambios en la trayectoria

    El conjunto de prueba nunca se modifica.
    """

    X_aumentado = [X_train]
    T_aumentado = [T_train]
    y_aumentado = [y_train]

    rng = np.random.default_rng(42)

    # Ruido pequeño en las secuencias.
    ruido_X = rng.normal(
        loc=0.0,
        scale=0.01,
        size=X_train.shape
    )

    X_variacion = X_train + ruido_X

    # Ruido pequeño en las trayectorias.
    ruido_T = rng.normal(
        loc=0.0,
        scale=0.01,
        size=T_train.shape
    )

    T_variacion = T_train + ruido_T

    X_aumentado.append(X_variacion)
    T_aumentado.append(T_variacion)
    y_aumentado.append(y_train.copy())

    X_final = np.concatenate(
        X_aumentado,
        axis=0
    )

    T_final = np.concatenate(
        T_aumentado,
        axis=0
    )

    y_final = np.concatenate(
        y_aumentado,
        axis=0
    )

    print(
        f"Aumento de datos activado: "
        f"{len(X_train)} -> {len(X_final)} ejemplos"
    )

    return X_final, T_final, y_final


def main():

    # ---------------------------------------------------------
    # 1. CARGAR DATASET
    # ---------------------------------------------------------

    (
        secuencias,
        trayectorias,
        etiquetas_texto,
        lista_senas
    ) = cargar_dataset()

    # ---------------------------------------------------------
    # 2. VERIFICAR FORMAS
    # ---------------------------------------------------------

    (
        secuencias,
        trayectorias,
        etiquetas_texto,
        longitud_trayectoria
    ) = verificar_formas(
        secuencias,
        trayectorias,
        etiquetas_texto
    )

    if len(secuencias) == 0:
        raise ValueError(
            "Después de verificar las formas no quedaron "
            "repeticiones válidas."
        )

    # ---------------------------------------------------------
    # 3. CONVERTIR A ARRAYS
    # ---------------------------------------------------------

    X = np.array(
        secuencias,
        dtype=np.float32
    )

    T = np.array(
        trayectorias,
        dtype=np.float32
    )

    # ---------------------------------------------------------
    # 4. CONVERTIR SEÑAS A NÚMEROS
    # ---------------------------------------------------------

    sena_a_indice = {
        seña: i
        for i, seña in enumerate(lista_senas)
    }

    y = np.array([
        sena_a_indice[etiqueta]
        for etiqueta in etiquetas_texto
    ])

    print(f"\nForma de X: {X.shape}")
    print(f"Forma de T: {T.shape}")
    print(f"Forma de y: {y.shape}")

    print(
        f"Longitud de trayectoria: "
        f"{longitud_trayectoria}"
    )

    print(
        f"Mapeo de señas a números: "
        f"{sena_a_indice}"
    )

    # ---------------------------------------------------------
    # 5. SPLIT ÚNICO
    # ---------------------------------------------------------

    indices = np.arange(len(X))

    indices_train, indices_test = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Los mismos índices se utilizan para ambos formatos.
    X_train = X[indices_train]
    X_test = X[indices_test]

    T_train = T[indices_train]
    T_test = T[indices_test]

    y_train = y[indices_train]
    y_test = y[indices_test]

    print(
        f"\nEjemplos de entrenamiento: "
        f"{len(X_train)}"
    )

    print(
        f"Ejemplos de prueba: "
        f"{len(X_test)}"
    )

    # ---------------------------------------------------------
    # 6. NORMALIZACIÓN
    # ---------------------------------------------------------

    (
        X_train,
        X_test,
        T_train,
        T_test,
        media_X,
        desviacion_X,
        media_T,
        desviacion_T
    ) = normalizar_datos(
        X_train,
        X_test,
        T_train,
        T_test
    )

    # ---------------------------------------------------------
    # 7. AUMENTO OPCIONAL
    # ---------------------------------------------------------

    if USAR_AUMENTO_DATOS:

        (
            X_train,
            T_train,
            y_train
        ) = aumentar_datos(
            X_train,
            T_train,
            y_train
        )

    # ---------------------------------------------------------
    # 8. GUARDAR DATOS
    # ---------------------------------------------------------

    np.savez_compressed(
        ARCHIVO_SALIDA,

        X_train=X_train,
        X_test=X_test,

        T_train=T_train,
        T_test=T_test,

        y_train=y_train,
        y_test=y_test,

        media_X=media_X,
        desviacion_X=desviacion_X,

        media_T=media_T,
        desviacion_T=desviacion_T
    )

    # ---------------------------------------------------------
    # 9. GUARDAR ETIQUETAS
    # ---------------------------------------------------------

    with open(
        ARCHIVO_ETIQUETAS,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            sena_a_indice,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ---------------------------------------------------------
    # 10. INFORMACIÓN FINAL
    # ---------------------------------------------------------

    print(
        f"\nListo. Datos guardados en "
        f"'{ARCHIVO_SALIDA}'"
    )

    print(
        f"Etiquetas guardadas en "
        f"'{ARCHIVO_ETIQUETAS}'"
    )

    print("\nContenido del archivo:")
    print("  X_train        -> secuencias de entrenamiento")
    print("  X_test         -> secuencias de prueba")
    print("  T_train        -> trayectorias de entrenamiento")
    print("  T_test         -> trayectorias de prueba")
    print("  y_train        -> etiquetas de entrenamiento")
    print("  y_test         -> etiquetas de prueba")
    print("  media_X        -> media de las secuencias")
    print("  desviacion_X   -> desviación de las secuencias")
    print("  media_T        -> media de las trayectorias")
    print("  desviacion_T   -> desviación de las trayectorias")


if __name__ == "__main__":
    main()