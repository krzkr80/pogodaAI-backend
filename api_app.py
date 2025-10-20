import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import json
from starlette.concurrency import run_in_threadpool

from loader import run_loader
from fix_gaps import fix_gaps
from train import train_model
from evaluate import evaluate_and_plot
from use import predict_one_day
from model_utils import GenericLSTM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScriptRequest(BaseModel):
    script_name: str
    parameters: Dict[str, Any] = {}

class LoadAndFilterRequest(BaseModel):
    hour: int = 6

class UseModelRequest(BaseModel):
    model_name: str
    input_data: list[float]

@app.post("/run_script/")
async def run_script(request: ScriptRequest):
    """
    Odbiera nazwę skryptu i parametry, a następnie bezpośrednio wykonuje odpowiednią funkcję Pythona.
    """
    script_name = request.script_name
    params = request.parameters

    if script_name == "train":
        print(f"INFO: Uruchamianie skryptu trenującego z parametrami: {params}", file=sys.stderr, flush=True)
        try:
            model_type = params.get("model_type")
            if not model_type:
                raise HTTPException(status_code=400, detail="Brak parametru 'model_type' dla skryptu trenującego.")
            
            variable_map = {
                "temperature": "TEMPERATURA_ŚRD",
                "pm25": "PM_25_ŚRD",
                "pm10": "PM_10_ŚRD",
                "cisnienie": "CIŚNIENIE_ŚRD",
                "wilgotnosc": "WILGOTNOŚĆ_ŚRD",
                "punkt_rosy": "PUNKT_ROSY_ŚRD"
            }
            variable_suffix = variable_map.get(model_type)
            if not variable_suffix:
                raise ValueError(f"Nieobsługiwany model_type: {model_type}. Nie znaleziono odpowiedniego przyrostka zmiennej.")

            training_metrics = await run_in_threadpool(
                train_model,
                model_type=model_type,
                input_path=params.get("input_path", "fixed_station_data.xlsx"),
                hidden_size=params.get("hidden_size", 64),
                num_layers=params.get("num_layers", 2),
                dropout=params.get("dropout", 0.2),
                learning_rate=params.get("learning_rate", 1e-3),
                epochs=params.get("epochs", 100),
                batch_size=params.get("batch_size", 32),
                variable_suffix=variable_suffix
            )
            
            print(f"INFO: Skrypt trenujący zakończył działanie pomyślnie. Metryki: {training_metrics}", file=sys.stderr, flush=True)
            return {
                "message": f"Skrypt '{script_name}' wykonany pomyślnie",
                "training_metrics": training_metrics
            }
        except Exception as e:
            import traceback
            print(f"BŁĄD: Wystąpił błąd podczas trenowania: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            raise HTTPException(status_code=500, detail=f"Wystąpił błąd podczas trenowania: {e}")

    elif script_name == "evaluate":
        print(f"INFO: Uruchamianie skryptu oceniającego z parametrami: {params}", file=sys.stderr, flush=True)
        try:
            plot_data_list = await run_in_threadpool(
                evaluate_and_plot,
                model_path=params.get("model_path"),
                data_path=params.get("data_path", "fixed_station_data.xlsx"),
                num_examples=params.get("num_examples", 5),
                station_filter=params.get("station", None),
                window_filter=params.get("window", None),
                enable_debug_logs=params.get("debug", False)
            )
            print(f"INFO: Skrypt oceniający zakończył działanie pomyślnie. Długość danych do wykresu: {len(plot_data_list) if plot_data_list else 0}", file=sys.stderr, flush=True)
            return {
                "message": f"Skrypt '{script_name}' wykonany pomyślnie",
                "plot_data": plot_data_list
            }
        except Exception as e:
            import traceback
            print(f"BŁĄD: Wystąpił błąd podczas oceny: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            raise HTTPException(status_code=500, detail=f"Wystąpił błąd podczas oceny: {e}")

    else:
        raise HTTPException(status_code=400, detail=f"Nieznany skrypt: {script_name}. Dozwolone skrypty: train, evaluate.")

@app.post("/load_and_filter/")
async def load_and_filter(request: LoadAndFilterRequest):
    """
    Wykonuje loader.run_loader, a następnie fix_gaps.fix_gaps z podanymi parametrami.
    """
    print(f"INFO: /load_and_filter/ otrzymał parametr godziny: {request.hour}", file=sys.stderr, flush=True)
    results = {}
    raw_data_output_file = "raw_combined_station_data.xlsx"
    fixed_data_output_file = "fixed_station_data.xlsx"

    try:
        print("INFO: Uruchamianie loader.run_loader...", file=sys.stderr, flush=True)
        await run_in_threadpool(run_loader, data_directory='data/', output_file=raw_data_output_file)
        results["loader_output"] = {"message": "loader.run_loader wykonany pomyślnie"}
        print("INFO: loader.run_loader zakończony.", file=sys.stderr, flush=True)

        print("INFO: Uruchamianie fix_gaps.fix_gaps...", file=sys.stderr, flush=True)
        await run_in_threadpool(fix_gaps, input_file=raw_data_output_file, output_file=fixed_data_output_file, start_hour=request.hour)
        results["fix_gaps_output"] = {"message": "fix_gaps.fix_gaps wykonany pomyślnie"}
        print("INFO: fix_gaps.fix_gaps zakończony.", file=sys.stderr, flush=True)

        return {
            "message": "Potok ładowania i filtrowania danych zakończony pomyślnie",
            **results
        }
    except Exception as e:
        import traceback
        print(f"BŁĄD: Potok danych nie powiódł się: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Potok danych nie powiódł się: {e}")

@app.post("/use/")
async def use_model(request: UseModelRequest):
    """
    Odbiera dane wejściowe modelu i zwraca prognozę przy użyciu określonego modelu.
    Dane wejściowe dla jednej stacji są replikowane dla wszystkich stacji, których oczekuje model,
    a zwracana jest średnia z prognoz.
    """
    if len(request.input_data) != 23:
        raise HTTPException(status_code=400, detail="Dane wejściowe muszą zawierać dokładnie 23 wartości godzinowe.")

    model_file_name = f"{request.model_name}_model.pt"
    model_path = model_file_name

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Nie znaleziono pliku modelu: {model_path}")

    # Zakodowane na stałe kolumny stacji, aby uniknąć ładowania torch
    STATION_COLUMNS_MAP = {
        "temperature": ['station_1048', 'station_1404', 'station_1709', 'station_1793', 'station_2044', 'station_2284', 'station_655'],
        "cisnienie": ['station_1048', 'station_1404', 'station_1709', 'station_1793', 'station_2044', 'station_2284', 'station_655'],
        "pm10": ['station_1048', 'station_1404', 'station_1709', 'station_1793', 'station_2044', 'station_2284', 'station_655'],
        "pm25": ['station_1048', 'station_1404', 'station_1709', 'station_1793', 'station_2044', 'station_2284', 'station_655'],
        "punkt_rosy": ['station_1048', 'station_1404', 'station_1709', 'station_1793', 'station_2044', 'station_2284', 'station_655'],
        "wilgotnosc": ['station_1048', 'station_1404', 'station_1709', 'station_1793', 'station_2044', 'station_2284', 'station_655'],
    }

    station_columns = STATION_COLUMNS_MAP.get(request.model_name)
    if not station_columns:
        raise HTTPException(status_code=404, detail=f"Identyfikatory stacji dla modelu '{request.model_name}' nie są zakodowane na stałe. Dodaj je do STATION_COLUMNS_MAP w api_app.py.")

    input_data_for_prediction = {}
    for station in station_columns:
        input_data_for_prediction[station] = request.input_data

    input_data_json_str = json.dumps(input_data_for_prediction)

    try:
        predictions = await run_in_threadpool(
            predict_one_day,
            model_path=model_path,
            input_data_json=input_data_json_str
        )
        
        if not predictions:
            raise HTTPException(status_code=500, detail="predict_one_day nie zwrócił żadnych prognoz.")
        
        total_prediction = sum(predictions.values())
        average_prediction = total_prediction / len(predictions)

        return {
            "message": f"Model '{request.model_name}' użyty pomyślnie",
            "average_prediction": average_prediction,
            "raw_predictions": predictions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prognoza nie powiodła się: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
