# Быстрый запуск проекта

## 1. Локальная подготовка

```bash
cd /Users/georgij/Projects/VKR3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Если окружение уже создано, достаточно:

```bash
cd /Users/georgij/Projects/VKR3
source .venv/bin/activate
```

## 2. Запуск ML-pipeline

```bash
export DVC_SITE_CACHE_DIR=.cache/dvc
dvc repro
```

После выполнения будут доступны:

```text
models/delay_model/model.pkl
models/reason_model/model.pkl
reports/metrics/delay_metrics.json
reports/metrics/reason_metrics.json
reports/figures/confusion_matrix_delay.png
reports/figures/confusion_matrix_reason.png
reports/figures/feature_importance_delay.png
reports/figures/feature_importance_reason.png
```

Полезные команды для защиты:

```bash
dvc dag
dvc status
dvc metrics show
dvc metrics diff
```

## 3. Запуск всей инфраструктуры

Перед запуском убедись, что Docker Desktop открыт.

```bash
docker compose up -d --build
```

Сервисы:

```text
FastAPI Swagger: http://localhost:8000/docs
FastAPI health:  http://localhost:8000/health
MLflow:          http://localhost:5001
Prometheus:      http://localhost:9090
Grafana:         http://localhost:3000
cAdvisor:        http://localhost:8080
```

Grafana:

```text
login: admin
password: admin
dashboard: Dashboards -> Flight Delay ML -> Flight Delay ML Demo
```

Prometheus targets:

```text
http://localhost:9090/targets
```

## 4. Записать эксперименты в MLflow внутри Docker

Если MLflow UI пустой после первого старта Docker Compose, запусти обучение через API:

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"dataset_id":"default","model_type":"random_forest","hyperparameters":{}}'
```

После завершения обнови `http://localhost:5001`.

## 5. Проверить прогноз

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "scheduled_departure_local": "2025-11-29 08:35",
    "airline_code": "FV",
    "origin_airport": "TJM",
    "destination_airport": "SVO",
    "aircraft_type": "A321",
    "distance_km": 1769,
    "planned_duration_min": 201,
    "weather_origin": "fog",
    "weather_destination": "clear",
    "temperature_origin_c": -3.9,
    "wind_speed_origin_mps": 3.8,
    "precipitation_origin_mm": 0.2,
    "visibility_origin_km": 1.9,
    "airport_load_index": 0.591,
    "airline_load_factor": 0.787,
    "previous_flight_delay_min": 38,
    "route_avg_delay_min": 6.5,
    "aircraft_age_years": 14,
    "technical_check_required": 0,
    "crew_change_required": 0
  }'
```

## 6. Остановка

```bash
docker compose down
```
