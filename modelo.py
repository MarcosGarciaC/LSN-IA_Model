"""
MODELO DE DOS RAMAS (usado por Entrenar.py y reconocer.py)
------------------------------------------------------------------
Rama 1 (secuencia): LSTM bidireccional sobre las 30 muestras. Usa la salida
en TODOS los pasos de tiempo, resumida como concat(promedio, maximo), asi
el inicio, el medio y el final del movimiento pesan igual.

Rama 2 (trayectoria): red densa pequeña sobre el vector de recorrido
completo de las muñecas.

Ambas se concatenan y pasan por un clasificador final.
"""

import torch
import torch.nn as nn


class ModeloDosRamas(nn.Module):
    def __init__(self, num_features, dim_trayectoria, num_clases,
                 tamano_oculto=64, num_capas=2, dropout=0.4,
                 tamano_trayectoria_oculto=32):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=tamano_oculto,
            num_layers=num_capas,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_capas > 1 else 0.0,
        )
        dim_salida_lstm = tamano_oculto * 2  # bidireccional

        self.rama_trayectoria = nn.Sequential(
            nn.Linear(dim_trayectoria, tamano_trayectoria_oculto),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        dim_concatenada = (dim_salida_lstm * 2) + tamano_trayectoria_oculto

        self.clasificador = nn.Sequential(
            nn.Linear(dim_concatenada, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_clases),
        )

    def forward(self, x, t):
        salida_lstm, _ = self.lstm(x)               # (lote, frames, 2*oculto)
        promedio = salida_lstm.mean(dim=1)
        maximo, _ = salida_lstm.max(dim=1)
        resumen_secuencia = torch.cat([promedio, maximo], dim=1)

        resumen_trayectoria = self.rama_trayectoria(t)

        combinado = torch.cat([resumen_secuencia, resumen_trayectoria], dim=1)
        return self.clasificador(combinado)
