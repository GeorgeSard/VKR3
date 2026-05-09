import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@dataclass
class ModelBundle:
    delay_model: Any | None
    reason_model: Any | None
    delay_metadata: dict[str, Any]
    reason_metadata: dict[str, Any]

    @property
    def is_loaded(self) -> bool:
        return self.delay_model is not None and self.reason_model is not None

    @property
    def model_version(self) -> dict[str, Any]:
        return {
            "delay_model": self.delay_metadata.get("version", "not_available"),
            "reason_model": self.reason_metadata.get("version", "not_available"),
        }


def _load_model(model_dir: Path) -> Any | None:
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def _load_metadata(model_dir: Path) -> dict[str, Any]:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_models(
    delay_model_dir: str | Path = "models/delay_model",
    reason_model_dir: str | Path = "models/reason_model",
) -> ModelBundle:
    delay_dir = Path(delay_model_dir)
    reason_dir = Path(reason_model_dir)
    return ModelBundle(
        delay_model=_load_model(delay_dir),
        reason_model=_load_model(reason_dir),
        delay_metadata=_load_metadata(delay_dir),
        reason_metadata=_load_metadata(reason_dir),
    )


def season_from_month(month: int | None) -> str | None:
    if month is None:
        return None
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _normalize_features(features: dict[str, Any], expected_columns: list[str]) -> pd.DataFrame:
    prepared = dict(features)

    if "airline_code" not in prepared and "airline" in prepared:
        prepared["airline_code"] = prepared["airline"]

    departure_value = prepared.get("scheduled_departure_local") or prepared.get(
        "scheduled_departure"
    )
    departure = pd.to_datetime(departure_value, errors="coerce") if departure_value else None
    if departure is not None and not pd.isna(departure):
        prepared.setdefault("flight_date", departure.date().isoformat())
        prepared.setdefault("departure_hour", int(departure.hour))
        prepared.setdefault("day_of_week", departure.day_name())
        prepared.setdefault("month", int(departure.month))
        prepared.setdefault("season", season_from_month(int(departure.month)))
        prepared.setdefault("is_weekend", int(departure.dayofweek >= 5))

    origin = prepared.get("origin_airport")
    destination = prepared.get("destination_airport")
    if "route" not in prepared and origin and destination:
        prepared["route"] = f"{str(origin).upper()}-{str(destination).upper()}"

    upper_columns = ["airline_code", "origin_airport", "destination_airport", "aircraft_type"]
    for column in upper_columns:
        if column in prepared and prepared[column] is not None:
            prepared[column] = str(prepared[column]).strip().upper()

    lower_columns = ["season", "weather_origin", "weather_destination"]
    for column in lower_columns:
        if column in prepared and prepared[column] is not None:
            prepared[column] = str(prepared[column]).strip().lower()

    if "day_of_week" in prepared and prepared["day_of_week"] is not None:
        prepared["day_of_week"] = str(prepared["day_of_week"]).strip().title()

    frame = pd.DataFrame([{column: prepared.get(column) for column in expected_columns}])
    return frame


def predict_flight(features: dict[str, Any], bundle: ModelBundle) -> dict[str, Any]:
    if bundle.delay_model is None:
        raise RuntimeError("Delay model is not loaded. Train the model first.")

    feature_columns = bundle.delay_metadata.get("feature_columns")
    if not feature_columns:
        raise RuntimeError("Delay model metadata does not contain feature_columns.")

    frame = _normalize_features(features, feature_columns)
    delay_probability = _positive_class_probability(bundle.delay_model, frame)
    threshold = float(bundle.delay_metadata.get("threshold", 0.5))
    is_significant_delay = delay_probability >= threshold

    predicted_reason = None
    reason_probability = None
    if is_significant_delay and bundle.reason_model is not None:
        reason_columns = bundle.reason_metadata.get("feature_columns", feature_columns)
        reason_frame = _normalize_features(features, reason_columns)
        predicted_reason = str(bundle.reason_model.predict(reason_frame)[0])
        reason_probability = _max_class_probability(bundle.reason_model, reason_frame)

    return {
        "is_significant_delay": bool(is_significant_delay),
        "delay_probability": round(float(delay_probability), 4),
        "predicted_reason": predicted_reason,
        "reason_probability": None
        if reason_probability is None
        else round(float(reason_probability), 4),
    }


def _positive_class_probability(model: Any, frame: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        classes = list(getattr(model, "classes_", []))
        positive_index = classes.index(1) if 1 in classes else len(probabilities) - 1
        return float(probabilities[positive_index])
    return float(model.predict(frame)[0])


def _max_class_probability(model: Any, frame: pd.DataFrame) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None
    return float(max(model.predict_proba(frame)[0]))
