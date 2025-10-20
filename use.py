import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse
import json
import io
import sys

from model_utils import GenericLSTM

def predict_one_day(
    model_path: str,
    input_data_json: str
):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    model_type = checkpoint.get('model_type', 'unknown')
    variable_suffix = checkpoint.get('variable_suffix', 'TEMPERATURA_ŚRD')
    n_stations = len(checkpoint["columns"])
    station_columns = checkpoint["columns"]

    model = GenericLSTM(n_stations=n_stations)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_data = json.loads(input_data_json)

    df_input = pd.DataFrame(index=range(23), columns=station_columns)
    for station in station_columns:
        if station in input_data:
            df_input[station] = input_data[station]
        else:
            raise ValueError(f"Brak danych dla stacji: {station} w danych wejściowych.")

    scaler_for_transform = StandardScaler()
    scaler_for_transform.mean_ = checkpoint["scaler_mean"]
    scaler_for_transform.scale_ = checkpoint["scaler_scale"]
    
    X_input_scaled = scaler_for_transform.transform(df_input.values)
    
    X_tensor = torch.tensor(X_input_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        preds_scaled = model(X_tensor).numpy()

    preds_orig = scaler_for_transform.inverse_transform(preds_scaled)

    result = {}
    for i, station in enumerate(station_columns):
        result[station] = preds_orig[0, i].item()

    print(json.dumps(result))
    return result
