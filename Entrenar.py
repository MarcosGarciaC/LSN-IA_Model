"""
PASO 3 : Entrenamiento con arquitectura de dos ramas
------------------------------------------------------------------------
Cambios respecto a la version anterior:
  - Recibe dos entradas por ejemplo: X (secuencia de landmarks, 30 frames)
    y T (vector de trayectoria resumido). Se procesan con dos ramas
    distintas de la red y se combinan antes de clasificar.
  - La rama de secuencia es una LSTM BIDIRECCIONAL que usa TODOS los
    pasos de tiempo (promedio + maximo), no solo el ultimo estado.
  - Menos capacidad (hidden=64) y mas regularizacion (dropout alto,
    weight_decay, ruido gaussiano en las entradas durante entrenamiento)
    para pelear contra el sobreajuste con pocos datos por sena.
  - Gradient clipping + scheduler de tasa de aprendizaje para estabilidad.
  - Seleccion de "mejor modelo" por un set de VALIDACION separado del
    set de entrenamiento (no por el set de prueba), para que la metrica
    final de prueba sea honesta y no este "contaminada" por haberse usado
    para elegir el checkpoint.
  - El checkpoint guarda todo lo necesario para reconstruir el modelo y
    normalizar igual durante el reconocimiento en vivo.


"""

import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# CONFIGURACION
# ============================================================
ARCHIVO_DATOS = "datos_preparados.npz"
ARCHIVO_ETIQUETAS = "etiquetas.json"
ARCHIVO_MODELO = "modelo_señas.pt"   

FRAMES_POR_SECUENCIA = 30

TAMANO_OCULTO = 64            # bajado de 128 -> 64 
NUM_CAPAS_LSTM = 2
DROPOUT = 0.4
TAMANO_TRAYECTORIA_OCULTO = 32

EPOCAS = 150
TASA_APRENDIZAJE = 1e-3
WEIGHT_DECAY = 1e-4
TAMANO_LOTE = 16               # subido de 8 -> 16
RUIDO_ENTRADA_STD = 0.01       # ruido gaussiano pequeno, solo en entrenamiento
PORCENTAJE_VALIDACION = 0.2    # se separa DEL SET DE ENTRENAMIENTO, no del de prueba
CLIP_GRAD_NORM = 1.0
PACIENCIA_SCHEDULER = 8        # epocas sin mejorar antes de bajar la tasa de aprendizaje


# ============================================================
# DATASET con dos entradas
# ============================================================
class DatasetSenas(Dataset):
    """Devuelve (X, T, y). Si ruido_std > 0, se agrega ruido gaussiano
    pequeno a X y T en cada acceso -- se usa SOLO en el dataset de
    entrenamiento, nunca en validacion ni en prueba."""

    def __init__(self, X, T, y, ruido_std=0.0):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.ruido_std = ruido_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        t = self.T[idx]
        if self.ruido_std > 0:
            x = x + torch.randn_like(x) * self.ruido_std
            t = t + torch.randn_like(t) * self.ruido_std
        return x, t, self.y[idx]


# ============================================================
# MODELO de dos ramas
# ============================================================
class ModeloDosRamas(nn.Module):
    """
    Rama 1 (secuencia): LSTM bidireccional sobre los 30 frames de
    landmarks. Se usa la salida en TODOS los pasos de tiempo, resumida
    como concat(promedio, maximo) sobre el eje temporal -- asi el
    inicio, el medio y el final del movimiento pesan igual.

    Rama 2 (trayectoria): red densa pequena que comprime el vector de
    trayectoria ya calculado en el Paso 2.

    Las dos ramas se concatenan y pasan por un clasificador final.
    """

    def __init__(self, num_features, dim_trayectoria, num_clases,
                 tamano_oculto=TAMANO_OCULTO, num_capas=NUM_CAPAS_LSTM,
                 dropout=DROPOUT, tamano_trayectoria_oculto=TAMANO_TRAYECTORIA_OCULTO):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=tamano_oculto,
            num_layers=num_capas,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_capas > 1 else 0.0,
        )
        dim_salida_lstm = tamano_oculto * 2  # x2 por ser bidireccional

        self.rama_trayectoria = nn.Sequential(
            nn.Linear(dim_trayectoria, tamano_trayectoria_oculto),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # x2 adicional porque concatenamos promedio Y maximo sobre el tiempo
        dim_concatenada = (dim_salida_lstm * 2) + tamano_trayectoria_oculto

        self.clasificador = nn.Sequential(
            nn.Linear(dim_concatenada, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_clases),
        )

    def forward(self, x, t):
        salida_lstm, _ = self.lstm(x)              # (lote, frames, 2*tamano_oculto)
        promedio = salida_lstm.mean(dim=1)           # resume todo el movimiento
        maximo, _ = salida_lstm.max(dim=1)           # captura el pico mas marcado
        resumen_secuencia = torch.cat([promedio, maximo], dim=1)

        resumen_trayectoria = self.rama_trayectoria(t)

        combinado = torch.cat([resumen_secuencia, resumen_trayectoria], dim=1)
        return self.clasificador(combinado)


# ============================================================
# Entrenamiento y evaluacion de una epoca
# ============================================================
def ejecutar_epoca(modelo, dataloader, funcion_perdida, dispositivo,
                    optimizador=None, entrenando=False):
    """Si optimizador no es None y entrenando=True, ajusta los pesos.
    Si no, solo evalua (usado para validacion y prueba)."""
    modelo.train() if entrenando else modelo.eval()

    perdida_total = 0.0
    correctos = 0
    total = 0

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
            predicciones = torch.argmax(salida, dim=1)
            correctos += (predicciones == lote_y).sum().item()
            total += lote_y.size(0)

    return perdida_total / total, correctos / total


def main():
    # --- Cargar datos preparados ---
    if not os.path.exists(ARCHIVO_DATOS):
        print(f"ERROR: no se encontro '{ARCHIVO_DATOS}'. Corre primero preparar_datos.py")
        return

    datos = np.load(ARCHIVO_DATOS)

    claves_requeridas = ["X_train", "X_test", "y_train", "y_test", "T_train", "T_test"]
    faltantes = [c for c in claves_requeridas if c not in datos]
    if faltantes:
        print(f"ERROR: faltan estas claves en '{ARCHIVO_DATOS}': {faltantes}")
        print("Revisa que tu preparar_datos.py (Paso 2) este guardando X_train/X_test, "
              "T_train/T_test y y_train/y_test con esos nombres exactos.")
        return

    X_train_full, X_test = datos["X_train"], datos["X_test"]
    T_train_full, T_test = datos["T_train"], datos["T_test"]
    y_train_full, y_test = datos["y_train"], datos["y_test"]

    with open(ARCHIVO_ETIQUETAS, "r", encoding="utf-8") as f:
        sena_a_indice = json.load(f)
    num_clases = len(sena_a_indice)

    num_features = X_train_full.shape[2]
    dim_trayectoria = T_train_full.shape[1]

    print(f"num_features (deberia ser 157): {num_features}")
    print(f"dim_trayectoria: {dim_trayectoria}")
    print(f"Senas: {list(sena_a_indice.keys())}")

    # --- Recuperar las 4 estadisticas de normalizacion  ---
    # Son 2 pares distintos porque X (secuencia) y T (trayectoria) tienen
    # escalas diferentes y se normalizaron por separado en preparar.py
    claves_normalizacion = ["media_X", "desviacion_X", "media_T", "desviacion_T"]
    faltan_norm = [c for c in claves_normalizacion if c not in datos]
    if faltan_norm:
        print(f"\nAVISO: faltan estas claves de normalizacion en '{ARCHIVO_DATOS}': {faltan_norm}")
        print("Sin ellas, el reconocimiento en vivo no podra normalizar igual que aqui.\n")
        media_X = desviacion_X = media_T = desviacion_T = None
    else:
        media_X = datos["media_X"]
        desviacion_X = datos["desviacion_X"]
        media_T = datos["media_T"]
        desviacion_T = datos["desviacion_T"]

    # --- Separar VALIDACION del set de ENTRENAMIENTO  ---
    # Asi el checkpoint se elige con datos que tampoco son el examen final,
    # y la precision de prueba que se reporta al final es honesta.
    X_train, X_val, T_train, T_val, y_train, y_val = train_test_split(
        X_train_full, T_train_full, y_train_full,
        test_size=PORCENTAJE_VALIDACION, random_state=42, stratify=y_train_full,
    )

    print(f"\nEntrenamiento: {len(X_train)}  |  Validacion: {len(X_val)}  |  Prueba: {len(X_test)}")

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dispositivo}")

    # --- DataLoaders ---
    ds_train = DatasetSenas(X_train, T_train, y_train, ruido_std=RUIDO_ENTRADA_STD)
    ds_val = DatasetSenas(X_val, T_val, y_val, ruido_std=0.0)
    ds_test = DatasetSenas(X_test, T_test, y_test, ruido_std=0.0)

    dl_train = DataLoader(ds_train, batch_size=TAMANO_LOTE, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=TAMANO_LOTE, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=TAMANO_LOTE, shuffle=False)

    # --- Modelo, perdida, optimizador, scheduler ---
    modelo = ModeloDosRamas(
        num_features=num_features,
        dim_trayectoria=dim_trayectoria,
        num_clases=num_clases,
    ).to(dispositivo)

    funcion_perdida = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(
        modelo.parameters(), lr=TASA_APRENDIZAJE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, mode="min", factor=0.5, patience=PACIENCIA_SCHEDULER
    )

    mejor_precision_val = 0.0

    print("\nIniciando entrenamiento...\n")
    for epoca in range(1, EPOCAS + 1):
        perdida_train, precision_train = ejecutar_epoca(
            modelo, dl_train, funcion_perdida, dispositivo, optimizador, entrenando=True
        )
        perdida_val, precision_val = ejecutar_epoca(
            modelo, dl_val, funcion_perdida, dispositivo, entrenando=False
        )

        scheduler.step(perdida_val)

        if epoca % 10 == 0 or epoca == 1:
            lr_actual = optimizador.param_groups[0]["lr"]
            print(f"Epoca {epoca:3d}/{EPOCAS} | "
                  f"Train perdida={perdida_train:.4f} acc={precision_train*100:.1f}% | "
                  f"Val perdida={perdida_val:.4f} acc={precision_val*100:.1f}% | "
                  f"lr={lr_actual:.2e}")

        if precision_val > mejor_precision_val:
            mejor_precision_val = precision_val
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
                "media_X": media_X,
                "desviacion_X": desviacion_X,
                "media_T": media_T,
                "desviacion_T": desviacion_T,
            }, ARCHIVO_MODELO)

    print(f"\nEntrenamiento terminado. Mejor precision de VALIDACION: {mejor_precision_val*100:.1f}%")

    # --- Evaluacion final, UNA SOLA VEZ, sobre el set de prueba ---
    checkpoint = torch.load(ARCHIVO_MODELO, map_location=dispositivo)
    modelo.load_state_dict(checkpoint["modelo_state_dict"])

    perdida_test, precision_test = ejecutar_epoca(
        modelo, dl_test, funcion_perdida, dispositivo, entrenando=False
    )
    print(f"Precision FINAL en PRUEBA (nunca vista durante entrenamiento ni seleccion): "
          f"{precision_test*100:.1f}%  (perdida={perdida_test:.4f})")

    # --- Matriz de confusion sobre el set de prueba ---
    indice_a_sena = {v: k for k, v in sena_a_indice.items()}
    nombres = [indice_a_sena[i] for i in range(num_clases)]

    modelo.eval()
    todas_predicciones = []
    with torch.no_grad():
        for lote_X, lote_T, lote_y in dl_test:
            lote_X, lote_T = lote_X.to(dispositivo), lote_T.to(dispositivo)
            salida = modelo(lote_X, lote_T)
            todas_predicciones.append(torch.argmax(salida, dim=1).cpu().numpy())
    y_pred = np.concatenate(todas_predicciones)

    etiquetas = list(range(num_clases))
    matriz = confusion_matrix(y_test, y_pred, labels=etiquetas)

    print("\nMATRIZ DE CONFUSION (filas = real, columnas = predicho)\n")
    ancho = max(len(n) for n in nombres) + 2
    print(" " * ancho + "".join(f"{n[:8]:>9}" for n in nombres))
    for i, nombre in enumerate(nombres):
        fila = "".join(f"{matriz[i][j]:>9}" for j in range(num_clases))
        print(f"{nombre:<{ancho}}{fila}")

    print("\nREPORTE POR SENA\n")
    print(classification_report(
        y_test, y_pred, labels=etiquetas, target_names=nombres, zero_division=0
    ))

    confusiones = []
    for i in range(num_clases):
        for j in range(num_clases):
            if i != j and matriz[i][j] > 0:
                confusiones.append((matriz[i][j], nombres[i], nombres[j]))
    confusiones.sort(reverse=True)

    if confusiones:
        print("PAREJAS DE SENAS QUE MAS SE CONFUNDEN:")
        for cantidad, real, predicho in confusiones[:5]:
            print(f"  '{real}' fue tomada como '{predicho}': {cantidad} vez/veces")
    else:
        print("No hubo confusiones en el set de prueba.")

    print(f"\nModelo guardado en: {ARCHIVO_MODELO}")


if __name__ == "__main__":
    main()