from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_params(params_path: str | Path = "params.yaml") -> dict[str, Any]:
    with open(params_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _day_name(date_series: pd.Series) -> pd.Series:
    return date_series.dt.day_name()


def make_features(params_path: str | Path = "params.yaml") -> Path:
    params = load_params(params_path)
    cleaned_path = Path(params["data"]["cleaned_path"])
    features_path = Path(params["data"]["features_path"])
    delay_threshold = params["target"]["delay_threshold_minutes"]
    feature_columns = params["features"]["categorical"] + params["features"]["numeric"]

    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {cleaned_path}")

    df = pd.read_csv(cleaned_path, encoding="utf-8-sig")
    df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")
    df["scheduled_departure_local"] = pd.to_datetime(
        df["scheduled_departure_local"], errors="coerce"
    )
    df = df.dropna(subset=["flight_date", "scheduled_departure_local"])

    df["departure_hour"] = df["scheduled_departure_local"].dt.hour
    df["day_of_week"] = _day_name(df["scheduled_departure_local"])
    df["month"] = df["scheduled_departure_local"].dt.month
    df["season"] = df["month"].apply(season_from_month)
    df["route"] = df["origin_airport"].astype(str) + "-" + df["destination_airport"].astype(str)
    df["is_significant_delay"] = (df["delay_minutes"] >= delay_threshold).astype(int)

    output_columns = [
        "flight_id",
        "flight_date",
        "scheduled_departure_local",
        "delay_minutes",
        "is_significant_delay",
        "delay_reason",
    ] + feature_columns
    missing_features = [column for column in output_columns if column not in df.columns]
    if missing_features:
        missing = ", ".join(missing_features)
        raise ValueError(f"Missing columns for feature dataset: {missing}")

    df = df[output_columns].copy()

    features_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_path, index=False)
    return features_path


def main() -> None:
    output_path = make_features()
    print(f"Feature dataset saved to {output_path}")


if __name__ == "__main__":
    main()
