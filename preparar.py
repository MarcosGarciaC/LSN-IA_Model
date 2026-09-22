"""
PASO 2: Preparar los datos grabados para entrenamiento
------------------------------------------------------------
Este script:
  1. Lee todas las carpetas dentro de dataset/ (cada carpeta = una seña)
  2. Carga todos los archivos .npy (cada uno = una repeticion grabada)
  3. Convierte los nombres de senas (texto) en numeros (esto lo necesita
     la red neuronal, no puede aprender directamente con texto)
  4. Divide todo en dos grupos: ENTRENAMIENTO (80%) y PRUEBA (20%)
  5. Guarda todo en un solo archivo listo para el siguiente paso


"""

import os
import json

import numpy as np
from sklearn.model_selection import train_test_split

CARPETA_DATASET = "dataset"
ARCHIVO_SALIDA = "datos_preparados.npz"
ARCHIVO_ETIQUETAS = "etiquetas.json"


def cargar_dataset():
    """Recorre dataset/<seña>/rep_XXX.npy y arma dos listas: secuencias y etiquetas."""
    if not os.path.exists(CARPETA_DATASET):
        raise FileNotFoundError(
            f"No existe la carpeta '{CARPETA_DATASET}'. Graba primero con grabar_dataset.py"
        )

    senas = sorted(os.listdir(CARPETA_DATASET))
    senas = [s for s in senas if os.path.isdir(os.path.join(CARPETA_DATASET, s))]

    if len(senas) == 0:
        raise ValueError("No hay ninguna carpeta de senas dentro de dataset/. Graba primero.")

    print(f"Senas encontradas: {senas}")

    secuencias = []
    etiquetas_texto = []

    for seña in senas:
        carpeta_sena = os.path.join(CARPETA_DATASET, seña)
        archivos = sorted([f for f in os.listdir(carpeta_sena) if f.endswith(".npy")])

        if len(archivos) == 0:
            print(f"  Aviso: '{seña}' no tiene ninguna repeticion grabada, se omite.")
            continue

        print(f"  {seña}: {len(archivos)} repeticiones")

        for archivo in archivos:
            ruta = os.path.join(carpeta_sena, archivo)
            secuencia = np.load(ruta)
            secuencias.append(secuencia)
            etiquetas_texto.append(seña)

    return secuencias, etiquetas_texto, senas


def verificar_forma_consistente(secuencias):
    """Todas las secuencias deben tener el mismo numero de frames y de valores
    por frame para poder convertirlas en un solo array. Si alguna quedo
    incompleta (por ejemplo la camara fallo un frame durante la grabacion),
    se avisa aqui en vez de fallar de forma confusa mas adelante."""
    formas = set(s.shape for s in secuencias)
    if len(formas) > 1:
        print("Aviso: hay secuencias con formas distintas:", formas)
        print("Se recortaran todas al tamano minimo comun para poder continuar.")
        min_frames = min(s.shape[0] for s in secuencias)
        secuencias = [s[:min_frames] for s in secuencias]
    return secuencias


def main():
    secuencias, etiquetas_texto, lista_senas = cargar_dataset()
    secuencias = verificar_forma_consistente(secuencias)

    X = np.array(secuencias)  # shape: (num_ejemplos, frames_por_secuencia, num_features)

    # Convertir nombres de senas a numeros: {"hola": 0, "gracias": 1, ...}
    sena_a_indice = {seña: i for i, seña in enumerate(lista_senas)}
    y = np.array([sena_a_indice[etiqueta] for etiqueta in etiquetas_texto])

    print(f"\nForma final de X (datos): {X.shape}")
    print(f"Forma final de y (etiquetas): {y.shape}")
    print(f"Mapeo de senas a numeros: {sena_a_indice}")

    # Dividir en entrenamiento (80%) y prueba (20%)
    # stratify=y asegura que cada seña quede representada proporcionalmente en ambos grupos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nEjemplos de entrenamiento: {len(X_train)}")
    print(f"Ejemplos de prueba: {len(X_test)}")

    # Guardar todo en un solo archivo comprimido
    np.savez_compressed(
        ARCHIVO_SALIDA,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
    )

    # Guardar el mapeo de senas a numeros (lo necesitaremos para el entrenamiento
    # y para mostrar el texto correcto durante el reconocimiento en vivo despues)
    with open(ARCHIVO_ETIQUETAS, "w", encoding="utf-8") as f:
        json.dump(sena_a_indice, f, ensure_ascii=False, indent=2)

    print(f"\nListo. Datos guardados en '{ARCHIVO_SALIDA}' y etiquetas en '{ARCHIVO_ETIQUETAS}'")


if __name__ == "__main__":
    main()