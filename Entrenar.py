"""
PASO 3: Entrenar el modelo LSTM
------------------------------------------------------------
Este script:
  1. Carga los datos ya preparados (datos_preparados.npz)
  2. Define una red neuronal LSTM (buena para leer secuencias en el tiempo,
     ideal para movimiento como las senas dinamicas)
  3. Entrena el modelo mostrandole los ejemplos varias veces (epocas)
  4. Evalua que tan bien predice con datos que nunca vio
  5. Guarda el modelo entrenado para usarlo despues en reconocimiento en vivo

No requiere camara. Puede tardar desde segundos hasta varios minutos
segun cuantos datos tengas y si tu PC tiene GPU o no (no es obligatorio
tener GPU para este tamano de datos).
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import Dataset, DataLoader

ARCHIVO_DATOS = "datos_preparados.npz"
ARCHIVO_ETIQUETAS = "etiquetas.json"
ARCHIVO_MODELO = "modelo_señas.pt"
ARCHIVO_MATRIZ = "matriz_confusion.png"

EPOCAS = 100
TASA_APRENDIZAJE = 0.001
TAMANO_LOTE = 8  # "batch size" - cuantos ejemplos ve el modelo a la vez


class DatasetSenas(Dataset):
    """Envuelve los arrays de numpy en un formato que PyTorch puede recorrer."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ModeloLSTM(nn.Module):
    """Red LSTM simple: lee la secuencia de landmarks frame por frame y al
    final decide a que sena corresponde."""

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
        # x tiene forma (lote, frames, features)
        salida_lstm, (h_n, c_n) = self.lstm(x)
        # Usamos solo el ultimo estado oculto, que resume toda la secuencia
        ultimo_estado = h_n[-1]
        return self.clasificador(ultimo_estado)


def main():
    # --- Cargar datos preparados ---
    datos = np.load(ARCHIVO_DATOS)
    X_train, y_train = datos["X_train"], datos["y_train"]
    X_test, y_test = datos["X_test"], datos["y_test"]

    with open(ARCHIVO_ETIQUETAS, "r", encoding="utf-8") as f:
        sena_a_indice = json.load(f)
    num_clases = len(sena_a_indice)
    num_features = X_train.shape[2]

    print(f"Datos de entrenamiento: {X_train.shape}")
    print(f"Datos de prueba: {X_test.shape}")
    print(f"Numero de senas distintas: {num_clases}")

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {dispositivo}")

    # --- Preparar DataLoaders ---
    ds_train = DatasetSenas(X_train, y_train)
    ds_test = DatasetSenas(X_test, y_test)
    dl_train = DataLoader(ds_train, batch_size=TAMANO_LOTE, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=TAMANO_LOTE, shuffle=False)

    # --- Crear modelo, funcion de perdida y optimizador ---
    modelo = ModeloLSTM(num_features=num_features, num_clases=num_clases).to(dispositivo)
    funcion_perdida = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)

    mejor_precision = 0.0

    print("\nIniciando entrenamiento...\n")
    for epoca in range(1, EPOCAS + 1):
        # --- Entrenamiento ---
        modelo.train()
        perdida_total = 0.0
        for lote_X, lote_y in dl_train:
            lote_X, lote_y = lote_X.to(dispositivo), lote_y.to(dispositivo)

            optimizador.zero_grad()
            salida = modelo(lote_X)
            perdida = funcion_perdida(salida, lote_y)
            perdida.backward()
            optimizador.step()

            perdida_total += perdida.item()

        # --- Evaluacion en el set de prueba ---
        modelo.eval()
        correctos = 0
        total = 0
        with torch.no_grad():
            for lote_X, lote_y in dl_test:
                lote_X, lote_y = lote_X.to(dispositivo), lote_y.to(dispositivo)
                salida = modelo(lote_X)
                predicciones = torch.argmax(salida, dim=1)
                correctos += (predicciones == lote_y).sum().item()
                total += lote_y.size(0)

        precision = correctos / total if total > 0 else 0.0

        if epoca % 10 == 0 or epoca == 1:
            print(f"Epoca {epoca:3d}/{EPOCAS} | Perdida: {perdida_total:.4f} | Precision en prueba: {precision*100:.1f}%")

        # Guardar el mejor modelo visto hasta ahora
        if precision > mejor_precision:
            mejor_precision = precision
            torch.save({
                "modelo_state_dict": modelo.state_dict(),
                "num_features": num_features,
                "num_clases": num_clases,
                "sena_a_indice": sena_a_indice,
            }, ARCHIVO_MODELO)

    print(f"\nEntrenamiento terminado. Mejor precision en prueba: {mejor_precision*100:.1f}%")
    print(f"Modelo guardado en: {ARCHIVO_MODELO}")

    checkpoint = torch.load(ARCHIVO_MODELO, map_location=dispositivo)
    modelo.load_state_dict(checkpoint["modelo_state_dict"])
    modelo.eval()
    predicciones_test = []
    with torch.no_grad():
        for lote_X, _ in dl_test:
            lote_X = lote_X.to(dispositivo)
            salida = modelo(lote_X)
            predicciones_test.extend(torch.argmax(salida, dim=1).cpu().numpy())

    matriz = confusion_matrix(y_test, predicciones_test, labels=np.arange(num_clases))
    nombres_senas = [
        sena for sena, _ in sorted(sena_a_indice.items(), key=lambda item: item[1])
    ]
    figura, eje = plt.subplots(figsize=(max(8, num_clases * 1.2), max(6, num_clases)))
    ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=nombres_senas,
    ).plot(ax=eje, cmap="Blues", values_format="d", xticks_rotation=45)
    figura.tight_layout()
    figura.savefig(ARCHIVO_MATRIZ, dpi=150)
    plt.close(figura)
    print(f"Matriz de confusion guardada en: {ARCHIVO_MATRIZ}")


if __name__ == "__main__":
    main()