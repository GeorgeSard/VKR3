from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_COLUMNS = {
    "flight_id",
    "flight_date",
    "scheduled_departure_local",
    "scheduled_arrival_local",
    "airline_code",
    "origin_airport",
    "destination_airport",
    "route",
    "delay_minutes",
    "delay_reason",
}


def load_params(params_path: str | Path = "params.yaml") -> dict[str, Any]:
    with open(params_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _clean_text(series: pd.Series, default: str = "unknown") -> pd.Series:
    return series.fillna(default).astype(str).str.strip()


def _fill_numeric_missing(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in df.columns:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
        median = df[column].median()
        df[column] = df[column].fillna(0 if pd.isna(median) else median)


def prepare_data(
    params_path: str | Path = "params.yaml",
    raw_path: str | Path | None = None,
) -> Path:
    params = load_params(params_path)
    raw_path = Path(raw_path or params["data"]["raw_path"])
    cleaned_path = Path(params["data"]["cleaned_path"])
    delay_threshold = int(params["target"]["delay_threshold_minutes"])

    if not raw_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {raw_path}")

    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["flight_id"], keep="first")

    df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")
    df["scheduled_departure_local"] = pd.to_datetime(
        df["scheduled_departure_local"], errors="coerce"
    )
    df["scheduled_arrival_local"] = pd.to_datetime(
        df["scheduled_arrival_local"], errors="coerce"
    )
    df["delay_minutes"] = pd.to_numeric(df["delay_minutes"], errors="coerce")

    text_columns = [
        "airline_code",
        "airline_name",
        "flight_number",
        "origin_airport",
        "origin_city",
        "destination_airport",
        "destination_city",
        "route",
        "aircraft_type",
        "day_of_week",
        "season",
        "weather_origin",
        "weather_destination",
        "delay_reason",
    ]
    for column in text_columns:
        if column in df.columns:
            df[column] = _clean_text(df[column])

    upper_columns = ["airline_code", "origin_airport", "destination_airport", "aircraft_type"]
    for column in upper_columns:
        if column in df.columns:
            df[column] = df[column].str.upper()

    lower_columns = ["season", "weather_origin", "weather_destination", "delay_reason"]
    for column in lower_columns:
        if column in df.columns:
            df[column] = df[column].str.lower()

    numeric_columns = [
        "distance_km",
        "planned_duration_min",
        "departure_hour",
        "month",
        "is_weekend",
        "temperature_origin_c",
        "wind_speed_origin_mps",
        "precipitation_origin_mm",
        "visibility_origin_km",
        "airport_load_index",
        "airline_load_factor",
        "previous_flight_delay_min",
        "route_avg_delay_min",
        "aircraft_age_years",
        "technical_check_required",
        "crew_change_required",
    ]
    _fill_numeric_missing(df, numeric_columns)

    df = df.dropna(
        subset=[
            "flight_id",
            "flight_date",
            "scheduled_departure_local",
            "delay_minutes",
        ]
    )
    df["route"] = df["origin_airport"] + "-" + df["destination_airport"]
    df["is_significant_delay"] = (
        df["delay_minutes"] >= delay_threshold
    ).astype(int)
    df.loc[df["is_significant_delay"] == 0, "delay_reason"] = "none"
    delayed_without_reason = (
        (df["is_significant_delay"] == 1)
        & df["delay_reason"].isin(["", "none", "unknown", "nan"])
    )
    df.loc[delayed_without_reason, "delay_reason"] = "other"

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_path, index=False)
    return cleaned_path


def main() -> None:
    output_path = prepare_data()
    print(f"Cleaned dataset saved to {output_path}")


if __name__ == "__main__":
    main()
