import os
import re
import pandas as pd
from typing import List, Dict

def find_station_files(directory: str) -> Dict[str, List[str]]:
    """
    Wyszukuje i grupuje pliki Excel według ID stacji na podstawie ich nazw.

    Args:
        directory: Katalog do przeszukania w poszukiwaniu plików .xlsx.

    Returns:
        Słownik mapujący ID stacji na listę odpowiadających im ścieżek plików.
    """
    station_files = {}
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx'):
            match = re.search(r'_(\d+)\.xlsx$', filename)
            if match:
                station_id = match.group(1)
                if station_id not in station_files:
                    station_files[station_id] = []
                station_files[station_id].append(os.path.join(directory, filename))
    return station_files

def load_station_data(file_paths: List[str]) -> pd.DataFrame:
    """
    Ładuje i łączy dane z listy plików Excel w jeden DataFrame.

    Args:
        file_paths: Lista ścieżek do plików Excel dla pojedynczej stacji.

    Returns:
        DataFrame biblioteki pandas z połączonymi danymi, posortowanymi według daty.
    """
    df_list = []
    for file_path in file_paths:
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            srd_columns = [col for col in df.columns if col.endswith('ŚRD')]
            if 'DATA' in df.columns and 'DATA' not in srd_columns:
                srd_columns.insert(0, 'DATA')
            
            df_list.append(df[srd_columns])
        except Exception as e:
            print(f"Ostrzeżenie: Nie można przetworzyć pliku {file_path}. Błąd: {e}")
    
    if not df_list:
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True)
    
    if 'DATA' in combined_df.columns:
        combined_df['DATA'] = pd.to_datetime(combined_df['DATA'])
        combined_df = combined_df.sort_values(by='DATA').reset_index(drop=True)
        
    return combined_df

def run_loader(data_directory: str = 'data/', output_file: str = 'raw_combined_station_data.xlsx'):
    """
    Funkcja uruchamiająca potok ładowania i przetwarzania danych.
    """
    station_files = find_station_files(data_directory)
    
    if not station_files:
        print(f"Nie znaleziono plików stacji w '{data_directory}'.")
        return

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for station_id, files in station_files.items():
            print(f"Ładowanie danych dla stacji {station_id}...")
            df = load_station_data(files)
            if not df.empty:
                df.to_excel(writer, sheet_name=f'station_{station_id}', index=False)
    
    print(f"\nWszystkie przetworzone dane zapisano w '{output_file}'.")
