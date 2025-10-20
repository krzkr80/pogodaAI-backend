import argparse
import pandas as pd
import numpy as np
import os

def get_block_start(timestamp: pd.Timestamp, start_hour: int) -> pd.Timestamp:
    """
    Określa początkowy znacznik czasu 24-godzinnego bloku (zaczynającego się o `start_hour`),
    do którego należy dany znacznik czasu.
    """
    if timestamp.hour < start_hour:
        return (timestamp - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=start_hour)
    else:
        return timestamp.normalize() + pd.Timedelta(hours=start_hour)

def fix_gaps(input_file: str, output_file: str, start_hour: int):
    """
    Naprawia luki w danych.
    - Przycinanie danych do dynamicznej daty początkowej i końcowej na podstawie godziny początkowej.
    - Interpolacja 1-godzinnych luk.
    - Usuwanie 24-godzinnych bloków danych wokół luk dłuższych niż 1 godzina.
    """
    try:
        station_data = pd.read_excel(input_file, sheet_name=None, engine='openpyxl')
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku wejściowego w '{input_file}'")
        return
    except Exception as e:
        print(f"Błąd podczas odczytu pliku wejściowego '{input_file}': {e}")
        return

    global_min_time = pd.Timestamp.max
    global_max_time = pd.Timestamp.min
    for df in station_data.values():
        if 'DATA' not in df.columns or df.empty:
            continue
        df['DATA'] = pd.to_datetime(df['DATA'])
        min_time = df['DATA'].min()
        max_time = df['DATA'].max()
        if min_time < global_min_time:
            global_min_time = min_time
        if max_time > global_max_time:
            global_max_time = max_time

    start_time = global_min_time.normalize() + pd.Timedelta(hours=start_hour)
    if global_min_time.hour >= start_hour:
        start_time += pd.Timedelta(days=1)

    end_hour = start_hour - 1 if start_hour > 0 else 23
    end_time = global_max_time.normalize() + pd.Timedelta(hours=end_hour)
    if global_max_time.hour < end_hour:
        end_time -= pd.Timedelta(days=1)
    
    print(f"Dane zostaną przycięte do początku o {start_time} i końca o {end_time}.")

    trimmed_station_data = {}
    for station_id, df in station_data.items():
        if 'DATA' in df.columns and not df.empty:
            df['DATA'] = pd.to_datetime(df['DATA'])
            trimmed_df = df[(df['DATA'] >= start_time) & (df['DATA'] <= end_time)].reset_index(drop=True)
            trimmed_station_data[station_id] = trimmed_df

    processed_station_data = {}
    for station_id, df in trimmed_station_data.items():
        if df.empty:
            continue
        df = df.drop_duplicates(subset=['DATA'])
        df = df.set_index('DATA').sort_index()
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        df = df.reindex(full_range)
        
        interpolated_df = df.interpolate(method='linear', limit=1, limit_direction='both')
        processed_station_data[station_id] = interpolated_df

    final_data = {}
    for station_id, df in processed_station_data.items():
        nan_timestamps = df[df.isnull().any(axis=1)].index

        if nan_timestamps.empty:
            final_data[station_id] = df.reset_index().rename(columns={'index': 'DATA'})
            continue

        blocks_to_remove = nan_timestamps.map(lambda ts: get_block_start(ts, start_hour)).unique()

        mask = ~df.index.map(lambda ts: get_block_start(ts, start_hour)).isin(blocks_to_remove)
        
        cleaned_df = df[mask].reset_index().rename(columns={'index': 'DATA'})
        
        data_columns = [col for col in cleaned_df.columns if col != 'DATA']
        cleaned_df.dropna(how='all', subset=data_columns, inplace=True)

        final_data[station_id] = cleaned_df

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for station_id, df in final_data.items():
            if not df.empty:
                df['DATA'] = pd.to_datetime(df['DATA'])
                trimmed_df = df[(df['DATA'] >= start_time) & (df['DATA'] <= end_time)].reset_index(drop=True)
                if not trimmed_df.empty:
                    trimmed_df.to_excel(writer, sheet_name=station_id, index=False)

    print(f"\nNaprawione dane zapisano w '{output_file}'.")

    try:
        xls_fixed = pd.ExcelFile(output_file)
        print("\nWeryfikacja naprawionych danych (pierwszy wiersz dla każdej stacji):")
        for sheet_name in xls_fixed.sheet_names:
            df_fixed = xls_fixed.parse(sheet_name)
            if not df_fixed.empty and 'DATA' in df_fixed.columns:
                print(f"Stacja: {sheet_name}, DATA pierwszego wiersza: {df_fixed['DATA'].iloc[0]}")
            elif df_fixed.empty:
                print(f"Stacja: {sheet_name}, Brak danych po naprawie.")
            else:
                print(f"Stacja: {sheet_name}, Nie znaleziono kolumny 'DATA' w naprawionych danych.")
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono naprawionego pliku wyjściowego w '{output_file}' do weryfikacji.")
    except Exception as e:
        print(f"Wystąpił błąd podczas weryfikacji: {e}")
