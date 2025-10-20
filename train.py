"""
train.py
------------------------------------
Trenuje model LSTM do prognozowania temperatury, PM2.5 lub PM10.

Wejście:
    fixed_station_data.xlsx
Wyjście:
    [model_type]_model.pt (wytrenowane wagi modelu)
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import argparse
import time
import json

from model_utils import load_data, build_tensors, GenericLSTM

def train_model(
    model_type: str,
    input_path: str = "fixed_station_data.xlsx",
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 32,
    variable_suffix: str = "TEMPERATURA_ŚRD"
):
    start_time = time.time()
    print(f"Rozpoczynanie trenowania modelu {model_type}...")
    df = load_data(input_path, variable_suffix)
    print(f"✅ Załadowano dane o kształcie: {df.shape}")

    X, y, scaler, timestamps = build_tensors(df)
    print(f"✅ Przygotowano tensory: X={X.shape}, y={y.shape}")

    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    model = GenericLSTM(
        n_stations=df.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    final_loss = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / len(train_loader)
        
    os.makedirs("plots", exist_ok=True)
    model_filename = f"{model_type}_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'columns': df.columns.tolist(),
        'variable_suffix': variable_suffix,
        'model_type': model_type,
        'split_idx': split_idx,
        'X_shape': X.shape,
        'y_shape': y.shape
    }, model_filename)

    end_time = time.time()
    time_taken = end_time - start_time

    return {"model_filename": model_filename, "epochs": epochs, "final_loss": final_loss, "time_taken": time_taken}
