Мне нужно создать учебный ML-проект для ВКР на тему:
«Разработка модели искусственного интеллекта для прогнозирования задержек авиарейсов и классификации их причин».

Цель проекта — не production-система, а демонстрация полного инженерного ML-процесса для защиты ВКР:
данные → версионирование данных → обработка → feature engineering → обучение моделей → сравнение метрик → подбор гиперпараметров → сохранение лучшей модели → FastAPI → мониторинг через Prometheus/Grafana.

Проект должен быть простым, понятным и удобным для демонстрации на защите.
Не нужно делать чрезмерно сложную production-архитектуру.

Проект должен решать две ML-задачи:

1. Прогноз операционно значимой задержки авиарейса.
   Это бинарная классификация.

   В рамках проекта под задержкой понимается не любое минимальное отклонение от расписания, а операционно значимая задержка.

   Правило формирования целевой переменной:
   delay_minutes >= 15 → significant_delay
   delay_minutes < 15 → no_significant_delay

   В README и комментариях нужно пояснить, что задержка меньше 15 минут не считается “идеально вовремя” в бытовом смысле, но для ML-задачи относится к классу небольших отклонений от расписания. Порог 15 минут выбран как распространённый операционный порог оценки пунктуальности авиарейсов.

2. Классификация причины задержки авиарейса.
   Это многоклассовая классификация.

   Модель классификации причины задержки должна применяться только для рейсов, у которых есть значимая задержка или у которых первая модель предсказала высокий риск значимой задержки.

Прогноз задержки в минутах пока НЕ реализовывать.
Регрессионную модель пока НЕ добавлять.
Метрики MAE, RMSE и R2 пока НЕ нужны.

Технологии:
- Python
- pandas
- scikit-learn
- XGBoost или LightGBM, если установка не усложняет проект
- MLflow для логирования экспериментов
- DVC для версионирования данных и pipeline
- FastAPI для API
- Docker Compose для запуска всех компонентов
- Prometheus + Grafana + cAdvisor для мониторинга контейнеров
- CSV как основной формат данных, потому что он человекочитаемый и удобен для демонстрации в Excel

Проект должен быть максимально простым и понятным.
Не добавлять Kubernetes.
Не добавлять Airflow.
Не добавлять unit-тесты.
Не добавлять сложную авторизацию.
Не добавлять лишние директории.
Код должен быть понятным новичку.
В коде должны быть комментарии.

Нужно сделать такую структуру проекта:

flight-delay-ml/
  app/
    main.py

  src/
    prepare_data.py
    make_features.py
    train_delay.py
    train_reason.py
    evaluate.py
    predict.py

  data/
    raw/
    interim/
    processed/

  models/
    delay_model/
    reason_model/

  reports/
    figures/
    metrics/

  monitoring/
    prometheus.yml
    grafana-dashboard.json

  params.yaml
  dvc.yaml
  Dockerfile
  docker-compose.yml
  requirements.txt
  README.md

Описание директорий и файлов:

app/
- директория FastAPI-приложения.

app/main.py:
- главный файл FastAPI;
- содержит API-ручки для демонстрации работы системы;
- не должен содержать всю ML-логику внутри себя;
- должен вызывать функции из src/.

Нужные FastAPI-ручки:

GET /health
- проверяет, что API работает;
- показывает, загружены ли модели;
- показывает текущую версию моделей.

POST /predict
- принимает признаки рейса;
- сначала вызывает модель прогноза значимой задержки;
- если модель предсказывает высокий риск значимой задержки, дополнительно вызывает модель классификации причины задержки;
- возвращает:
  is_significant_delay,
  delay_probability,
  predicted_reason,
  reason_probability.

Пример ответа:
{
  "is_significant_delay": true,
  "delay_probability": 0.78,
  "predicted_reason": "weather",
  "reason_probability": 0.65
}

Если значимая задержка не прогнозируется, predicted_reason можно вернуть как null.

POST /datasets/upload
- принимает CSV-файл;
- сохраняет его как новый датасет;
- возвращает dataset_id;
- данные должны быть пригодны для дальнейшего запуска pipeline.

POST /train
- запускает обучение моделей;
- принимает параметры:
  dataset_id,
  model_type,
  гиперпараметры;
- запускает обучение модели прогноза значимой задержки и модели классификации причины;
- логирует результаты в MLflow;
- возвращает run_id или список run_id для двух моделей.

GET /experiments
- возвращает список экспериментов из MLflow:
  название запуска,
  тип задачи,
  тип модели,
  версия данных,
  основные метрики,
  время запуска.

GET /runs/{run_id}
- возвращает подробную информацию по конкретному MLflow run:
  параметры,
  метрики,
  путь к артефактам,
  версия данных,
  версия признаков.

GET /system/containers
- показывает состояние Docker-контейнеров;
- для учебной демонстрации можно использовать docker socket;
- в README обязательно указать, что такой подход используется только локально для учебной демонстрации.

src/
- директория с ML-логикой проекта.

src/prepare_data.py:
- читает data/raw/flights.csv;
- удаляет дубли;
- обрабатывает пропуски;
- приводит даты и время к нормальному формату;
- проверяет наличие нужных столбцов;
- проверяет наличие delay_minutes;
- проверяет наличие delay_reason для задачи классификации причин;
- сохраняет очищенные данные в data/interim/cleaned_flights.csv.

src/make_features.py:
- читает data/interim/cleaned_flights.csv;
- создаёт признаки для моделей:
  departure_hour,
  day_of_week,
  month,
  season,
  airline,
  origin_airport,
  destination_airport,
  route;
- создаёт целевую переменную для первой задачи:
  is_significant_delay = delay_minutes >= 15;
- подготавливает target delay_reason для второй задачи;
- для модели причин оставляет только рейсы со значимой задержкой и известной причиной;
- сохраняет итоговый датасет в data/processed/features.csv.

src/train_delay.py:
- обучает модель прогноза операционно значимой задержки;
- задача: binary classification;
- целевой признак: is_significant_delay;
- данные делятся на train / validation / test;
- train используется для обучения модели;
- validation используется для выбора модели, гиперпараметров и threshold;
- test используется только для финальной честной оценки;
- желательно использовать time_based split, чтобы модель обучалась на прошлых рейсах и проверялась на более новых;
- считает метрики:
  precision,
  recall,
  f1-score,
  ROC-AUC,
  PR-AUC,
  confusion matrix;
- основной метрикой считать F1-score;
- accuracy не использовать как основную метрику, потому что при дисбалансе классов она может быть misleading;
- сохраняет модель в models/delay_model/;
- логирует параметры, метрики, модель и артефакты в MLflow.

src/train_reason.py:
- обучает модель классификации причины задержки;
- задача: multiclass classification;
- модель должна обучаться только на рейсах со значимой задержкой и известной причиной;
- целевой признак: delay_reason;
- данные делятся на train / validation / test;
- считает метрики:
  macro-F1,
  weighted-F1,
  confusion matrix;
- основной метрикой считать macro-F1, потому что классы причин задержек могут быть несбалансированы;
- сохраняет модель в models/reason_model/;
- логирует параметры, метрики, модель и артефакты в MLflow.

src/evaluate.py:
- содержит общие функции оценки моделей;
- считает метрики;
- строит confusion matrix;
- сохраняет метрики в reports/metrics/;
- сохраняет графики в reports/figures/;
- используется из train_delay.py и train_reason.py.

src/predict.py:
- загружает обученную модель прогноза значимой задержки;
- загружает обученную модель классификации причины задержки;
- принимает признаки рейса;
- сначала прогнозирует вероятность значимой задержки;
- если вероятность выше заданного threshold, прогнозирует причину задержки;
- возвращает результат для FastAPI.

data/
- директория с данными проекта.

data/raw/
- исходные данные;
- основной файл: flights.csv;
- данные должны храниться в CSV-формате.

data/interim/
- промежуточные очищенные данные;
- пример файла: cleaned_flights.csv.

data/processed/
- готовые данные для обучения моделей;
- пример файла: features.csv.

models/
- директория для сохранённых моделей.

models/delay_model/
- сохранённая модель прогноза операционно значимой задержки;
- можно хранить model.pkl и metadata.json.

models/reason_model/
- сохранённая модель классификации причины задержки;
- можно хранить model.pkl и metadata.json.

reports/
- директория с отчётами, графиками и метриками.

reports/figures/
- графики:
  confusion_matrix_delay.png,
  confusion_matrix_reason.png,
  feature_importance_delay.png,
  feature_importance_reason.png,
  model_comparison.png.

reports/metrics/
- JSON-файлы с метриками:
  delay_metrics.json,
  reason_metrics.json.

monitoring/
- директория с настройками мониторинга.

monitoring/prometheus.yml:
- конфигурация Prometheus;
- Prometheus должен собирать метрики с api и cadvisor.

monitoring/grafana-dashboard.json:
- готовый dashboard для Grafana;
- нужен для демонстрации состояния контейнеров и API;
- на dashboard желательно показать CPU, RAM, состояние контейнеров и базовые метрики API.

params.yaml:
- файл с параметрами проекта;
- параметры не должны быть зашиты прямо в коде.

Пример содержимого params.yaml:

data:
  raw_path: data/raw/flights.csv
  cleaned_path: data/interim/cleaned_flights.csv
  features_path: data/processed/features.csv

split:
  strategy: time_based
  train_size: 0.7
  val_size: 0.15
  test_size: 0.15
  random_state: 42

target:
  delay_threshold_minutes: 15

delay_model:
  model_type: random_forest
  n_estimators: 200
  max_depth: 10
  threshold: 0.5

reason_model:
  model_type: random_forest
  n_estimators: 200
  max_depth: 10

dvc.yaml:
- описывает DVC pipeline;
- pipeline должен состоять из этапов:
  prepare,
  featurize,
  train_delay,
  train_reason.

DVC должен отслеживать:
- исходные данные;
- очищенные данные;
- обработанные признаки;
- параметры;
- метрики;
- сохранённые модели.

Примерная логика dvc.yaml:

stages:
  prepare:
    cmd: python src/prepare_data.py
    deps:
      - data/raw/flights.csv
      - src/prepare_data.py
    outs:
      - data/interim/cleaned_flights.csv

  featurize:
    cmd: python src/make_features.py
    deps:
      - data/interim/cleaned_flights.csv
      - src/make_features.py
    outs:
      - data/processed/features.csv

  train_delay:
    cmd: python src/train_delay.py
    deps:
      - data/processed/features.csv
      - src/train_delay.py
      - src/evaluate.py
    params:
      - split
      - target
      - delay_model
    outs:
      - models/delay_model
    metrics:
      - reports/metrics/delay_metrics.json

  train_reason:
    cmd: python src/train_reason.py
    deps:
      - data/processed/features.csv
      - src/train_reason.py
      - src/evaluate.py
    params:
      - split
      - reason_model
    outs:
      - models/reason_model
    metrics:
      - reports/metrics/reason_metrics.json

Dockerfile:
- собирает контейнер для FastAPI и ML-кода;
- должен устанавливать зависимости из requirements.txt;
- должен запускать FastAPI через uvicorn.

docker-compose.yml:
- запускает весь проект одной командой:
  docker compose up -d

В docker-compose.yml должны быть сервисы:
- api
- mlflow
- prometheus
- grafana
- cadvisor

api:
- FastAPI-приложение;
- должно иметь доступ к данным, моделям и MLflow.

mlflow:
- MLflow Tracking Server;
- нужен для просмотра экспериментов, параметров, метрик и моделей.

prometheus:
- собирает метрики.

grafana:
- показывает dashboard.

cadvisor:
- собирает метрики Docker-контейнеров.

requirements.txt:
- зависимости Python:
  fastapi
  uvicorn
  pandas
  scikit-learn
  mlflow
  dvc
  python-multipart
  prometheus-client
  docker
  matplotlib
  pyyaml
  joblib
  xgboost или lightgbm, если используется

README.md:
- должно содержать:
  1. Название темы ВКР.
  2. Цель проекта.
  3. Описание двух ML-задач.
  4. Объяснение порога 15 минут.
  5. Схему ML-процесса.
  6. Описание структуры проекта.
  7. Команды запуска Docker Compose.
  8. Команды DVC.
  9. Команды MLflow.
  10. Примеры запросов к FastAPI.
  11. Описание метрик.
  12. Список скриншотов для защиты.

MLflow должен логировать:

Для модели прогноза значимой задержки:
- task_type = significant_delay_prediction
- model_type
- гиперпараметры
- dataset_version
- split_strategy
- feature_set_version
- delay_threshold_minutes
- precision
- recall
- f1
- ROC-AUC
- PR-AUC
- confusion matrix как artifact
- feature importance как artifact
- обученную модель

Для модели классификации причины задержки:
- task_type = delay_reason_classification
- model_type
- гиперпараметры
- dataset_version
- split_strategy
- feature_set_version
- macro-F1
- weighted-F1
- confusion matrix как artifact
- feature importance как artifact
- обученную модель

DVC должен позволять показать:
- какие данные были исходными;
- какие данные получились после очистки;
- какие признаки были сформированы;
- на какой версии данных обучалась модель;
- какие параметры были изменены;
- какие метрики получились после запуска pipeline.

Важно:
- использовать CSV как основной формат данных;
- не использовать Parquet как основной формат;
- проект должен быть удобен для демонстрации на защите;
- все ключевые результаты должны быть видны через DVC, MLflow, FastAPI и Grafana;
- архитектура должна быть простой;
- не добавлять лишних директорий и файлов;
- не усложнять проект без необходимости.

Также подготовь README так, чтобы по нему можно было сделать скриншоты для защиты:

Скриншоты для защиты:
1. Структура проекта в VS Code.
2. Исходный CSV-файл.
3. Очищенный CSV-файл.
4. DVC pipeline.
5. DVC status.
6. DVC metrics diff.
7. MLflow UI со списком экспериментов.
8. MLflow run с гиперпараметрами.
9. MLflow run с метриками.
10. Confusion matrix для прогноза значимой задержки.
11. Confusion matrix для классификации причины задержки.
12. Feature importance.
13. Swagger UI FastAPI.
14. POST /predict.
15. POST /train.
16. GET /experiments.
17. Grafana dashboard.
18. Prometheus targets.
19. Docker Compose containers.
