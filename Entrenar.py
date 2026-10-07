"""
PASO 3: Entrenar el modelo de dos ramas
------------------------------------------------------------------------
  - Carga datos_preparados.npz (entrenamiento / validacion / prueba ya
    separados y normalizados en el Paso 2).
  - Aumento de datos al vuelo SOLO en entrenamiento: ruido gaussiano y
    pequeños desplazamientos en el tiempo (la seña puede empezar un poco
    antes o despues dentro de la ventana).
  - El mejor checkpoint se elige con VALIDACION; la prueba se evalua una
    sola vez al final, asi la metrica es honesta.
  - Regularizacion: dropout, weight_decay, label smoothing, gradient
    clipping, scheduler de tasa de aprendizaje y parada temprana.
  - El checkpoint guarda todo lo necesario para reconstruir el modelo y
    normalizar igual en reconocer.py (estadisticas como tensores, por lo
    que torch.load funciona sin banderas especiales).
"""

import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from caracteristicas import (
    FRAMES_POR_SECUENCIA, FPS_SECUENCIA, NUM_FEATURES_FRAME, DIM_TRAYECTORIA,
)
from modelo import ModeloDosRamas

# ============================================================
# CONFIGURACION
# ============================================================
ARCHIVO_DATOS = "datos_preparados.npz"
ARCHIVO_ETIQUETAS = "etiquetas.json"
ARCHIVO_MODELO = "modelo_señas.pt"
ARCHIVO_MATRIZ = "matriz_confusion.png"

TAMANO_OCULTO = 64
NUM_CAPAS_LSTM = 2
DROPOUT = 0.4
TAMANO_TRAYECTORIA_OCULTO = 32

EPOCAS = 150
TASA_APRENDIZAJE = 1e-3
WEIGHT_DECAY = 1e-4
TAMANO_LOTE = 16
LABEL_SMOOTHING = 0.05
CLIP_GRAD_NORM = 1.0
PACIENCIA_SCHEDULER = 8
PACIENCIA_PARADA = 40          # epocas sin mejorar la validacion antes de parar

# Aumento de datos (solo entrenamiento). Los datos ya estan normalizados,
# asi que 0.05 equivale a un 5% de una desviacion estandar.
RUIDO_ENTRADA_STD = 0.05
DESPLAZAMIENTO_MAX = 2         # muestras (±), con repeticion del borde

SEMILLA = 42


class DatasetSenas(Dataset):
    def __init__(self, X, T, y, aumentar=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.aumentar = aumentar

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        t = self.T[idx]
        if self.aumentar:
            desp = random.randint(-DESPLAZAMIENTO_MAX, DESPLAZAMIENTO_MAX)
            if desp != 0:
                pasos = torch.clamp(torch.arange(x.shape[0]) + desp, 0, x.shape[0] - 1)
                x = x[pasos]
            if RUIDO_ENTRADA_STD > 0:
                x = x + torch.randn_like(x) * RUIDO_ENTRADA_STD
                t = t + torch.randn_like(t) * RUIDO_ENTRADA_STD
        return x, t, self.y[idx]


def ejecutar_epoca(modelo, dataloader, funcion_perdida, dispositivo,
                   optimizador=None, entrenando=False):
    modelo.train() if entrenando else modelo.eval()

    perdida_total, correctos, total = 0.0, 0, 0
    contexto = torch.enable_grad() if entrenando else torch.no_grad()
    with contexto:
        for lote_X, lote_T, lote_y in dataloader:
            lote_X = lote_X.to(dispositivo)
            lote_T = lote_T.to(dispositivo)
            lote_y = lote_y.to(dispositivo)

            if entrenando:
                optimizador.zero_grad()

            salida = modelo(lote_X, lote_T)
            perdida = funcion_perdida(salida, lote_y)

            if entrenando:
                perdida.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), CLIP_GRAD_NORM)
                optimizador.step()

            perdida_total += perdida.item() * lote_y.size(0)
            correctos += (torch.argmax(salida, dim=1) == lote_y).sum().item()
            total += lote_y.size(0)

    return perdida_total / total, correctos / total


def main():
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)

    if not os.path.exists(ARCHIVO_DATOS):
        print(f"ERROR: no se encontro '{ARCHIVO_DATOS}'. Corre primero preparar.py")
        return

    datos = np.load(ARCHIVO_DATOS)
    claves = ["X_train", "X_val", "X_test", "T_train", "T_val", "T_test",
              "y_train", "y_val", "y_test",
              "media_X", "desviacion_X", "media_T", "desviacion_T"]
    faltantes = [c for c in claves if c not in datos]
    if faltantes:
        print(f"ERROR: faltan claves en '{ARCHIVO_DATOS}': {faltantes}")
        print("Vuelve a correr preparar.py (version nueva).")
        return

    X_train, X_val, X_test = datos["X_train"], datos["X_val"], datos["X_test"]
    T_train, T_val, T_test = datos["T_train"], datos["T_val"], datos["T_test"]
    y_train, y_val, y_test = datos["y_train"], datos["y_val"], datos["y_test"]

    with open(ARCHIVO_ETIQUETAS, "r", encoding="utf-8") as f:
        sena_a_indice = json.load(f)
    num_clases = len(sena_a_indice)

    num_features = X_train.shape[2]
    dim_trayectoria = T_train.shape[1]
    if num_features != NUM_FEATURES_FRAME or dim_trayectoria != DIM_TRAYECTORIA:
        print("ERROR: las dimensiones de los datos no coinciden con caracteristicas.py "
              f"({num_features}/{dim_trayectoria} vs {NUM_FEATURES_FRAME}/{DIM_TRAYECTORIA}). "
              "Regraba o vuelve a correr preparar.py.")
        return

    print(f"Entrenamiento: {len(X_train)} | Validacion: {len(X_val)} | Prueba: {len(X_test)}")
    print(f"Señas: {num_clases} | num_features: {num_features} | dim_trayectoria: {dim_trayectoria}")

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dispositivo}")

    dl_train = DataLoader(DatasetSenas(X_train, T_train, y_train, aumentar=True),
                          batch_size=TAMANO_LOTE, shuffle=True)
    dl_val = DataLoader(DatasetSenas(X_val, T_val, y_val), batch_size=TAMANO_LOTE, shuffle=False)
    dl_test = DataLoader(DatasetSenas(X_test, T_test, y_test), batch_size=TAMANO_LOTE, shuffle=False)

    modelo = ModeloDosRamas(
        num_features=num_features,
        dim_trayectoria=dim_trayectoria,
        num_clases=num_clases,
        tamano_oculto=TAMANO_OCULTO,
        num_capas=NUM_CAPAS_LSTM,
        dropout=DROPOUT,
        tamano_trayectoria_oculto=TAMANO_TRAYECTORIA_OCULTO,
    ).to(dispositivo)

    funcion_perdida = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    # Para validacion/prueba se reporta la perdida sin suavizado
    perdida_eval = nn.CrossEntropyLoss()

    optimizador = torch.optim.Adam(
        modelo.parameters(), lr=TASA_APRENDIZAJE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, mode="min", factor=0.5, patience=PACIENCIA_SCHEDULER
    )

    mejor = (-1.0, -float("inf"))   # (precision_val, -perdida_val)
    epocas_sin_mejorar = 0

    print("\nIniciando entrenamiento...\n")
    for epoca in range(1, EPOCAS + 1):
        perdida_train, precision_train = ejecutar_epoca(
            modelo, dl_train, funcion_perdida, dispositivo, optimizador, entrenando=True
        )
        perdida_val, precision_val = ejecutar_epoca(
            modelo, dl_val, perdida_eval, dispositivo, entrenando=False
        )
        scheduler.step(perdida_val)

        if epoca % 10 == 0 or epoca == 1:
            lr_actual = optimizador.param_groups[0]["lr"]
            print(f"Epoca {epoca:3d}/{EPOCAS} | "
                  f"Train perdida={perdida_train:.4f} acc={precision_train*100:.1f}% | "
                  f"Val perdida={perdida_val:.4f} acc={precision_val*100:.1f}% | "
                  f"lr={lr_actual:.2e}")

        candidato = (precision_val, -perdida_val)
        if candidato > mejor:
            mejor = candidato
            epocas_sin_mejorar = 0
            torch.save({
                "modelo_state_dict": modelo.state_dict(),
                "num_features": num_features,
                "dim_trayectoria": dim_trayectoria,
                "num_clases": num_clases,
                "sena_a_indice": sena_a_indice,
                "tamano_oculto": TAMANO_OCULTO,
                "num_capas_lstm": NUM_CAPAS_LSTM,
                "dropout": DROPOUT,
                "tamano_trayectoria_oculto": TAMANO_TRAYECTORIA_OCULTO,
                "frames_por_secuencia": FRAMES_POR_SECUENCIA,
                "fps_secuencia": FPS_SECUENCIA,
                # Estadisticas como tensores: torch.load las lee sin problemas
                "media_X": torch.tensor(datos["media_X"]),
                "desviacion_X": torch.tensor(datos["desviacion_X"]),
                "media_T": torch.tensor(datos["media_T"]),
                "desviacion_T": torch.tensor(datos["desviacion_T"]),
            }, ARCHIVO_MODELO)
        else:
            epocas_sin_mejorar += 1
            if epocas_sin_mejorar >= PACIENCIA_PARADA:
                print(f"\nParada temprana en la epoca {epoca}: "
                      f"{PACIENCIA_PARADA} epocas sin mejorar la validacion.")
                break

    print(f"\nEntrenamiento terminado. Mejor precision de VALIDACION: {mejor[0]*100:.1f}%")

    # --- Evaluacion final, UNA SOLA VEZ, sobre prueba ---
    checkpoint = torch.load(ARCHIVO_MODELO, map_location=dispositivo)
    modelo.load_state_dict(checkpoint["modelo_state_dict"])

    perdida_test, precision_test = ejecutar_epoca(
        modelo, dl_test, perdida_eval, dispositivo, entrenando=False
    )
    print(f"Precision FINAL en PRUEBA: {precision_test*100:.1f}%  (perdida={perdida_test:.4f})")

    # --- Matriz de confusion y reporte ---
    indice_a_sena = {v: k for k, v in sena_a_indice.items()}
    nombres = [indice_a_sena[i] for i in range(num_clases)]

    modelo.eval()
    predicciones = []
    with torch.no_grad():
        for lote_X, lote_T, _ in dl_test:
            salida = modelo(lote_X.to(dispositivo), lote_T.to(dispositivo))
            predicciones.append(torch.argmax(salida, dim=1).cpu().numpy())
    y_pred = np.concatenate(predicciones)

    etiquetas = list(range(num_clases))
    matriz = confusion_matrix(y_test, y_pred, labels=etiquetas)

    print("\nREPORTE POR SEÑA\n")
    print(classification_report(
        y_test, y_pred, labels=etiquetas, target_names=nombres, zero_division=0
    ))

    confusiones = [(matriz[i][j], nombres[i], nombres[j])
                   for i in range(num_clases) for j in range(num_clases)
                   if i != j and matriz[i][j] > 0]
    confusiones.sort(reverse=True)
    if confusiones:
        print("PAREJAS DE SEÑAS QUE MAS SE CONFUNDEN:")
        for cantidad, real, predicho in confusiones[:8]:
            print(f"  '{real}' fue tomada como '{predicho}': {cantidad} vez/veces")
    else:
        print("No hubo confusiones en el set de prueba.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        figura, eje = plt.subplots(figsize=(max(8, num_clases * 0.7), max(6, num_clases * 0.6)))
        ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=nombres).plot(
            ax=eje, cmap="Blues", values_format="d", xticks_rotation=60, colorbar=False
        )
        figura.tight_layout()
        figura.savefig(ARCHIVO_MATRIZ, dpi=150)
        plt.close(figura)
        print(f"\nMatriz de confusion guardada en: {ARCHIVO_MATRIZ}")
    except ImportError:
        print("\n(matplotlib no esta instalado: se omite la imagen de la matriz)")

    print(f"Modelo guardado en: {ARCHIVO_MODELO}")


if __name__ == "__main__":
    main()