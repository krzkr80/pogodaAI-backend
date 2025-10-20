import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import os

def load_data(filepath: str, variable_suffix: str) -> pd.DataFrame:
    """Wczytuje wszystkie arkusze stacji i łączy określone kolumny zmiennych."""
    xls = pd.ExcelFile(filepath)
    data_frames = []
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        if 'DATA' not in df.columns:
            continue
        df['DATA'] = pd.to_datetime(df['DATA'])
        cols = [c for c in df.columns if variable_suffix in c]
        if not cols:
            print(f"⚠️ Nie znaleziono kolumny '{variable_suffix}' w {sheet_name}")
            continue
        df = df[['DATA'] + cols]
        df = df.rename(columns={cols[0]: sheet_name})
        data_frames.append(df)

    if not data_frames:
        raise ValueError(f"Nie znaleziono danych '{variable_suffix}' w skoroszycie.")

    merged = data_frames[0]
    for df in data_frames[1:]:
        merged = pd.merge(merged, df, on="DATA", how="outer")

    merged = merged.sort_values("DATA").set_index("DATA")
    return merged

def build_tensors(df: pd.DataFrame):
    """Tworzy kroczące 24-godzinne okna do predykcji szeregów czasowych."""
    df = df.sort_index().interpolate().dropna()
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

    X, y, timestamps = [], [], []
    
    for i in range(0, len(df_scaled) - 24, 24):
        window = df_scaled.iloc[i:i+24]
        
        time_diff = window.index[-1] - window.index[0]
        if time_diff == pd.Timedelta(hours=23):
            X.append(window.iloc[:23].values)
            y.append(window.iloc[23].values)
            timestamps.append(window.index.strftime('%Y-%m-%d %H:%M').tolist())

    if not X:
        raise ValueError("Nie można utworzyć żadnych 24-godzinnych okien z danych. Sprawdź ciągłość danych.")
        
    X = np.stack(X)
    y = np.stack(y)
    return X, y, scaler, timestamps

class GenericLSTM(nn.Module):
    def __init__(self, n_stations: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_stations,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, n_stations)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out
