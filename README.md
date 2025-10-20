# Backend PogodaAI

Ten backend służy do przetwarzania danych pogodowych, trenowania modeli predykcyjnych i serwowania prognoz za pośrednictwem API.

## Jak uruchomić

1.  **Zainstaluj zależności:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Uruchom serwer API:**
    ```bash
    uvicorn api_app:app --reload
    ```
    Serwer będzie dostępny pod adresem `http://127.0.0.1:8000`.

## Punkty końcowe API (Endpoints)

Wszystkie żądania należy wysyłać jako `POST` z ciałem w formacie JSON.

---

### 1. `/load_and_filter/`

Uruchamia pełny potok przetwarzania danych: wczytuje surowe pliki Excel z katalogu `data/`, łączy je, a następnie czyści i uzupełnia brakujące dane.

**Żądanie (Request):**

```json
{
  "hour": 6
}
```

*   `hour` (integer, opcjonalnie, domyślnie `6`): Godzina, od której mają zaczynać się 24-godzinne bloki danych.

**Odpowiedź (Response):**

```json
{
  "message": "Potok ładowania i filtrowania danych zakończony pomyślnie",
  "loader_output": {
    "message": "loader.run_loader wykonany pomyślnie"
  },
  "fix_gaps_output": {
    "message": "fix_gaps.fix_gaps wykonany pomyślnie"
  }
}
```

---

### 2. `/run_script/`

Umożliwia uruchomienie określonych skryptów, takich jak `train` (trenowanie) lub `evaluate` (ocena).

#### Trenowanie modelu (`train`)

**Żądanie (Request):**

```json
{
  "script_name": "train",
  "parameters": {
    "model_type": "temperature",
    "epochs": 150
  }
}
```

*   `model_type` (string, wymagane): Typ modelu do wytrenowania. Dostępne opcje: `"temperature"`, `"pm25"`, `"pm10"`, `"cisnienie"`, `"wilgotnosc"`, `"punkt_rosy"`.
*   Pozostałe parametry (np. `hidden_size`, `num_layers`, `epochs`) są opcjonalne i mają wartości domyślne.

**Odpowiedź (Response):**

```json
{
  "message": "Skrypt 'train' wykonany pomyślnie",
  "training_metrics": {
    "model_filename": "temperature_model.pt",
    "epochs": 150,
    "final_loss": 0.0123,
    "time_taken": 123.45
  }
}
```

#### Ocena modelu (`evaluate`)

**Żądanie (Request):**

```json
{
  "script_name": "evaluate",
  "parameters": {
    "model_path": "temperature_model.pt",
    "num_examples": 3
  }
}
```

*   `model_path` (string, wymagane): Ścieżka do wytrenowanego modelu (`.pt`).
*   `num_examples` (integer, opcjonalnie): Liczba losowych przykładów do oceny.

**Odpowiedź (Response):**

```json
{
  "message": "Skrypt 'evaluate' wykonany pomyślnie",
  "plot_data": [
    {
      "model_type": "temperature",
      "station": "station_1048",
      "window_idx": 42,
      "unit": "°C",
      "timestamps": ["..."],
      "past_hours_data": [...],
      "real_value": 15.5,
      "predicted_value": 15.2
    }
  ]
}
```

---

### 3. `/use/`

Wykorzystuje wytrenowany model do wygenerowania prognozy na podstawie 23 godzin danych wejściowych.

**Żądanie (Request):**

```json
{
  "model_name": "temperature",
  "input_data": [
    10.1, 10.2, 10.3, 10.5, 10.6, 10.8, 11.0, 11.2, 11.5, 12.0, 12.3, 12.5,
    12.8, 13.0, 13.1, 13.0, 12.8, 12.5, 12.2, 11.9, 11.5, 11.2, 11.0
  ]
}
```

*   `model_name` (string, wymagane): Nazwa modelu do użycia (np. `"temperature"`).
*   `input_data` (lista 23 floatów, wymagane): Sekwencja 23 kolejnych godzinnych pomiarów.

**Odpowiedź (Response):**

```json
{
  "message": "Model 'temperature' użyty pomyślnie",
  "average_prediction": 10.85,
  "raw_predictions": {
    "station_1048": 10.9,
    "station_1404": 10.8
  }
}
