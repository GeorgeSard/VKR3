from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from starlette.responses import Response

from src.make_features import make_features
from src.predict import load_models, predict_flight
from src.prepare_data import load_params, prepare_data
from src.train_delay import train_delay_model
from src.train_reason import train_reason_model


APP_VERSION = "0.1.0"
DATA_RAW_DIR = Path("data/raw")
PARAMS_PATH = Path("params.yaml")

REQUEST_COUNTER = Counter(
    "flight_delay_api_requests_total",
    "Total number of API requests.",
    ["endpoint"],
)

app = FastAPI(
    title="Flight Delay ML API",
    description="Educational API for flight delay prediction demo.",
    version=APP_VERSION,
)

model_bundle = None


class FlightFeatures(BaseModel):
    scheduled_departure_local: str | None = None
    airline_code: str | None = None
    origin_airport: str | None = None
    destination_airport: str | None = None
    aircraft_type: str | None = None
    distance_km: float | None = None
    planned_duration_min: float | None = None
    weather_origin: str | None = None
    weather_destination: str | None = None
    temperature_origin_c: float | None = None
    wind_speed_origin_mps: float | None = None
    precipitation_origin_mm: float | None = None
    visibility_origin_km: float | None = None
    airport_load_index: float | None = None
    airline_load_factor: float | None = None
    previous_flight_delay_min: float | None = None
    route_avg_delay_min: float | None = None
    aircraft_age_years: float | None = None
    technical_check_required: int | None = None
    crew_change_required: int | None = None

    class Config:
        extra = "allow"


class TrainRequest(BaseModel):
    dataset_id: str = "default"
    model_type: str = "random_forest"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


@app.on_event("startup")
def startup() -> None:
    global model_bundle
    model_bundle = load_models()


@app.get("/health")
def health() -> dict[str, Any]:
    REQUEST_COUNTER.labels(endpoint="/health").inc()
    bundle = model_bundle or load_models()
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "models_loaded": bundle.is_loaded,
        "model_version": bundle.model_version,
    }


@app.post("/predict")
def predict(features: FlightFeatures) -> dict[str, Any]:
    REQUEST_COUNTER.labels(endpoint="/predict").inc()
    bundle = model_bundle or load_models()
    feature_payload = (
        features.model_dump() if hasattr(features, "model_dump") else features.dict()
    )
    try:
        return predict_flight(feature_payload, bundle)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, str]:
    REQUEST_COUNTER.labels(endpoint="/datasets/upload").inc()
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    dataset_id = f"flights_{timestamp}"
    output_path = DATA_RAW_DIR / f"{dataset_id}.csv"
    output_path.write_bytes(await file.read())

    return {"dataset_id": dataset_id, "path": str(output_path)}


def _resolve_dataset_path(dataset_id: str) -> Path:
    params = load_params(PARAMS_PATH)
    default_path = Path(params["data"]["raw_path"])
    if dataset_id in {"default", "current", default_path.stem}:
        return default_path

    candidate = DATA_RAW_DIR / dataset_id
    if candidate.suffix.lower() != ".csv":
        candidate = candidate.with_suffix(".csv")
    if candidate.exists():
        return candidate

    raise HTTPException(
        status_code=404,
        detail=f"Dataset '{dataset_id}' was not found in {DATA_RAW_DIR}.",
    )


@app.post("/train")
def train_models(request: TrainRequest) -> dict[str, Any]:
    global model_bundle
    REQUEST_COUNTER.labels(endpoint="/train").inc()
    if request.model_type != "random_forest":
        raise HTTPException(
            status_code=400,
            detail="Only random_forest is supported in the educational version.",
        )

    dataset_path = _resolve_dataset_path(request.dataset_id)
    prepare_data(PARAMS_PATH, raw_path=dataset_path)
    make_features(PARAMS_PATH)

    delay_result = train_delay_model(PARAMS_PATH, request.hyperparameters)
    reason_result = train_reason_model(PARAMS_PATH, request.hyperparameters)
    model_bundle = load_models()
    return {
        "dataset_id": request.dataset_id,
        "runs": [
            {"task_type": delay_result["task_type"], "run_id": delay_result["run_id"]},
            {"task_type": reason_result["task_type"], "run_id": reason_result["run_id"]},
        ],
        "metrics": {
            "delay": delay_result["metrics"],
            "reason": reason_result["metrics"],
        },
    }


@app.get("/experiments")
def experiments() -> dict[str, Any]:
    REQUEST_COUNTER.labels(endpoint="/experiments").inc()
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="MLflow is not installed.") from exc

    client = MlflowClient()
    result = []
    for experiment in client.search_experiments():
        runs = client.search_runs(
            [experiment.experiment_id],
            max_results=20,
            order_by=["attributes.start_time DESC"],
        )
        for run in runs:
            result.append(
                {
                    "run_id": run.info.run_id,
                    "experiment_name": experiment.name,
                    "run_name": run.data.tags.get("mlflow.runName"),
                    "task_type": run.data.tags.get("task_type"),
                    "model_type": run.data.params.get("delay_model.model_type")
                    or run.data.params.get("reason_model.model_type"),
                    "dataset_version": run.data.tags.get("dataset_version"),
                    "metrics": run.data.metrics,
                    "start_time": datetime.fromtimestamp(
                        run.info.start_time / 1000, tz=timezone.utc
                    ).isoformat()
                    if run.info.start_time
                    else None,
                }
            )
    return {"experiments": result}


@app.get("/runs/{run_id}")
def run_details(run_id: str) -> dict[str, Any]:
    REQUEST_COUNTER.labels(endpoint="/runs/{run_id}").inc()
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="MLflow is not installed.") from exc

    client = MlflowClient()
    try:
        run = client.get_run(run_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "run_id": run.info.run_id,
        "status": run.info.status,
        "artifact_uri": run.info.artifact_uri,
        "params": run.data.params,
        "metrics": run.data.metrics,
        "tags": run.data.tags,
    }


@app.get("/system/containers")
def system_containers() -> dict[str, Any]:
    REQUEST_COUNTER.labels(endpoint="/system/containers").inc()
    try:
        import docker

        client = docker.from_env()
        containers = [
            {
                "name": container.name,
                "status": container.status,
                "image": container.image.tags,
            }
            for container in client.containers.list(all=True)
        ]
        return {"available": True, "containers": containers}
    except Exception as exc:
        return {
            "available": False,
            "message": "Docker socket is available only in local demo mode.",
            "error": str(exc),
        }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
