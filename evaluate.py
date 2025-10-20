"""
evaluate.py
-----------------------------
Ocenia wytrenowany model LSTM i rysuje przykładowe prognozy.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import os
import argparse
import io
import sys
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from model_utils import load_data, build_tensors, GenericLSTM

def evaluate_and_plot(
    model_path: str,
    data_path: str = "fixed_station_data.xlsx",
    num_examples: int = 5,
    station_filter: str = None,
    window_filter: int = None,
    enable_debug_logs: bool = False
):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    model_type = checkpoint.get('model_type', 'unknown')
    variable_suffix = checkpoint.get('variable_suffix', 'TEMPERATURA_ŚRD')
    n_stations = len(checkpoint["columns"])
    split_idx = checkpoint.get('split_idx', None)
    X_shape = checkpoint.get('X_shape', None)
    y_shape = checkpoint.get('y_shape', None)

    if split_idx is None or X_shape is None or y_shape is None:
        raise ValueError("W punkcie kontrolnym modelu brakuje 'split_idx', 'X_shape' lub 'y_shape'. Proszę ponownie wytrenować model za pomocą zaktualizowanego skryptu `train.py`.")

    df = load_data(data_path, variable_suffix)
    X, y, scaler_rebuilt, all_timestamps = build_tensors(df)

    if X.shape != X_shape or y.shape != y_shape:
        print(f"Ostrzeżenie: Kształt przebudowanych danych X={X.shape}, y={y.shape} nie pasuje do zapisanego w modelu X_shape={X_shape}, y_shape={y_shape}. Może to prowadzić do nieprawidłowej oceny.")
    
    X_test, y_test = X[split_idx:], y[split_idx:]
    test_timestamps = all_timestamps[split_idx:]

    model = GenericLSTM(n_stations=n_stations)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    scaler_for_inverse = StandardScaler()
    scaler_for_inverse.mean_ = checkpoint["scaler_mean"]
    scaler_for_inverse.scale_ = checkpoint["scaler_scale"]
    preds_orig = scaler_for_inverse.inverse_transform(preds)
    y_test_orig = scaler_for_inverse.inverse_transform(y_test.reshape(-1, y_test.shape[1]))

    if window_filter is not None:
        if 0 <= window_filter < len(X_test):
            days_to_process = [window_filter]
        else:
            print(f"⚠️ Okno {window_filter} jest poza zakresem zestawu testowego (0-{len(X_test)-1}). Wykresy nie zostaną wygenerowane.")
            days_to_process = []
    else:
        days_to_process = random.sample(range(len(X_test)), min(num_examples, len(X_test)))

    unit = ""
    if model_type == "temperature":
        unit = "°C"
    elif model_type == "pm25" or model_type == "pm10":
        unit = "µg/m³"
    elif model_type == "cisnienie":
        unit = "hPa"
    elif model_type == "wilgotnosc":
        unit = "%"
    elif model_type == "punkt_rosy":
        unit = "°C"

    plot_data_list = []

    for day_idx in days_to_process:
        for s, station in enumerate(checkpoint["columns"]):
            if station_filter is not None and station_filter != "all" and station_filter != station:
                continue

            seq = X_test[day_idx, :, s] * scaler_for_inverse.scale_[s] + scaler_for_inverse.mean_[s]
            real_next = y_test_orig[day_idx, s]
            pred_next = preds_orig[day_idx, s]

            if enable_debug_logs:
                print(f"\n--- Debugowanie: Stacja {station}, Okno {day_idx} ---")
                print(f"Poprzednie 23 godziny (przeskalowane i odwrócone): {seq[:23].tolist()}")
                print(f"Rzeczywista wartość (24. godzina): {real_next}")
                print(f"Przewidywana wartość (24. godzina): {pred_next}")
                print("--------------------------------------------------")

            plot_data = {
                "model_type": model_type,
                "station": station,
                "window_idx": day_idx,
                "unit": unit,
                "timestamps": test_timestamps[day_idx],
                "past_hours_data": seq[:23].tolist(),
                "real_value": real_next.item(),
                "predicted_value": pred_next.item(),
            }
            plot_data_list.append(plot_data)
    
    return plot_data_list
