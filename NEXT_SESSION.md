# Напоминалка для следующей сессии

## Текущее состояние

- Репозиторий: `/Users/georgij/Projects/VKR3`
- Remote: `https://github.com/GeorgeSard/VKR3.git`
- Последний запушенный commit на `main`: `39da2a8 Record Docker training outputs`
- Docker Compose уже был поднят и проверен:
  - FastAPI: `http://localhost:8000/docs`
  - MLflow: `http://localhost:5001`
  - Grafana: `http://localhost:3000`, `admin/admin`
  - Prometheus: `http://localhost:9090`
  - cAdvisor: `http://localhost:8080`
- MLflow runs после Docker-обучения:
  - delay model: `8b5b2bd80a954be5b9559b55e6d6a43c`
  - reason model: `ace4b669a14144dda447adf19e7ca3e5`
- `dvc status` был чистый: `Data and pipelines are up to date`.
- Внешний порт MLflow специально `5001`, потому что на macOS порт `5000` занят `ControlCenter`.

## Что пользователь хочет дальше

Нужно подготовить демонстрационный сценарий для скриншотов ВКР:

1. Показать, что сначала модель обучалась на одних/менее очищенных данных.
2. Потом данные были очищены.
3. После очистки модель обучилась на другой версии данных.
4. Через DVC показать, что поменялись данные, pipeline outputs и метрики.
5. Затем менять гиперпараметры модели и показывать, как меняются метрики.

Главная цель: не только получить лучшую модель, а получить понятные скриншоты:

- `dvc dag`
- `dvc status`
- `dvc metrics show`
- `dvc metrics diff`
- MLflow runs с разными параметрами
- confusion matrix
- feature importance
- Grafana/Prometheus/API

## Рекомендуемый план реализации

### 1. Сделать режимы очистки данных

В `params.yaml` добавить секцию:

```yaml
cleaning:
  mode: full
  drop_duplicates: true
  fill_missing_numeric: median
  fill_missing_categorical: unknown
```

В `src/prepare_data.py` реализовать два режима:

- `baseline`:
  - минимальная очистка;
  - не удалять дубли по `flight_id`;
  - пропуски заполнять простыми значениями;
  - сохранить в `data/interim/cleaned_flights.csv`;
  - цель: получить "до очистки" как первую версию pipeline.
- `full`:
  - текущая нормальная очистка;
  - удалить дубли;
  - привести даты;
  - заполнить пропуски медианой/`unknown`;
  - пересчитать `is_significant_delay`;
  - привести `delay_reason`.

Важно: не подавать в признаки `delay_minutes`, `is_significant_delay`, `delay_reason`.

### 2. Зафиксировать baseline-эксперимент

Команды:

```bash
source .venv/bin/activate
export DVC_SITE_CACHE_DIR=.cache/dvc

dvc exp run -S cleaning.mode=baseline
dvc exp show
dvc metrics show
```

Если нужен обычный Git/DVC diff для скриншотов, можно сделать отдельный commit:

```bash
git checkout -b demo/baseline-cleaning
dvc repro
git add params.yaml dvc.lock reports/figures reports/metrics
git commit -m "Train baseline cleaning version"
```

Потом перейти обратно на `main` и сделать cleaned/full:

```bash
git checkout main
dvc repro
git add params.yaml dvc.lock reports/figures reports/metrics
git commit -m "Train full cleaned data version"
dvc metrics diff demo/baseline-cleaning main
```

### 3. Гиперпараметры для скриншотов

Использовать DVC experiments:

```bash
dvc exp run -S delay_model.n_estimators=100 -S delay_model.max_depth=6
dvc exp run -S delay_model.n_estimators=300 -S delay_model.max_depth=12
dvc exp run -S delay_model.threshold=0.35
dvc exp show
dvc metrics diff
```

Можно также запускать через API, чтобы runs появлялись в MLflow:

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"dataset_id":"default","model_type":"random_forest","hyperparameters":{"n_estimators":300,"max_depth":12}}'
```

Но для DVC-скриншотов лучше сначала использовать `dvc exp run`, потому что он явно показывает изменение params/metrics.

### 4. Что проверить после каждого запуска

```bash
dvc status
dvc metrics show
dvc metrics diff
curl http://localhost:8000/health
curl http://localhost:8000/experiments
```

Ожидаемые ключевые метрики:

- delay model:
  - `f1`
  - `precision`
  - `recall`
  - `roc_auc`
  - `pr_auc`
- reason model:
  - `macro_f1`
  - `weighted_f1`

## Осторожно

- `reports/metrics/delay_metrics.json` и `reports/metrics/reason_metrics.json` сейчас должны управляться DVC, не Git.
- `data/raw/flight_delays_ru_synthetic_2023_2025.csv` управляется DVC через `.dvc`-файл, не Git.
- `models/delay_model` и `models/reason_model` игнорируются Git и являются DVC outputs/local artifacts.
- Не коммитить `.venv`, `.cache`, `.dvc/cache`, `mlflow/`, `mlflow.db`, `mlruns/`.
- Если Docker build снова станет огромным, проверить `.dockerignore`.
- Если MLflow снова ругается на Host header, в `docker-compose.yml` уже должен быть `--allowed-hosts "*"`.

## Быстрый старт перед продолжением

```bash
cd /Users/georgij/Projects/VKR3
source .venv/bin/activate
export DVC_SITE_CACHE_DIR=.cache/dvc
docker compose ps
dvc status
git status --short --branch
```

