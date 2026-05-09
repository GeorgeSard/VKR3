import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from src.evaluate import (
        multiclass_classification_metrics,
        save_confusion_matrix,
        save_feature_importance,
        save_metrics,
    )
except ModuleNotFoundError:
    from evaluate import (
        multiclass_classification_metrics,
        save_confusion_matrix,
        save_feature_importance,
        save_metrics,
    )


def load_params(params_path: str | Path = "params.yaml") -> dict[str, Any]:
    with open(params_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def time_based_split(
    df: pd.DataFrame,
    date_column: str,
    train_size: float,
    val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sorted_df = df.sort_values(date_column).reset_index(drop=True)
    train_end = int(len(sorted_df) * train_size)
    val_end = train_end + int(len(sorted_df) * val_size)
    return (
        sorted_df.iloc[:train_end].copy(),
        sorted_df.iloc[train_end:val_end].copy(),
        sorted_df.iloc[val_end:].copy(),
    )


def build_model(
    categorical_features: list[str],
    numeric_features: list[str],
    settings: dict[str, Any],
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=int(settings.get("n_estimators", 200)),
        max_depth=settings.get("max_depth"),
        class_weight=settings.get("class_weight", "balanced"),
        random_state=int(settings.get("random_state", 42)),
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def model_feature_names(model: Pipeline) -> list[str]:
    preprocessor = model.named_steps["preprocessor"]
    return [str(name) for name in preprocessor.get_feature_names_out()]


def log_to_mlflow(
    run_name: str,
    model: Pipeline,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[Path],
    tags: dict[str, str],
) -> str | None:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("MLflow is not installed; metrics were saved locally only.")
        return None

    try:
        mlflow.set_experiment("flight-delay-vkr")
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(tags)
            for section, values in params.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        mlflow.log_param(f"{section}.{key}", value)
                else:
                    mlflow.log_param(section, values)

            for key, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    mlflow.log_metric(key, float(value))

            for artifact in artifacts:
                if artifact.exists():
                    mlflow.log_artifact(str(artifact))

            mlflow.sklearn.log_model(model, artifact_path="model")
            return run.info.run_id
    except Exception as exc:
        print(f"MLflow logging failed: {exc}")
        return None


def train_reason_model(
    params_path: str | Path = "params.yaml",
    model_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = load_params(params_path)
    settings = dict(params["reason_model"])
    if model_overrides:
        settings.update(model_overrides)
    settings["random_state"] = params["split"]["random_state"]

    features_path = Path(params["data"]["features_path"])
    model_dir = Path("models/reason_model")
    metrics_path = Path("reports/metrics/reason_metrics.json")
    confusion_matrix_path = Path("reports/figures/confusion_matrix_reason.png")
    feature_importance_csv = Path("reports/metrics/feature_importance_reason.csv")
    feature_importance_png = Path("reports/figures/feature_importance_reason.png")

    df = pd.read_csv(features_path, encoding="utf-8-sig")
    df[params["data"]["date_column"]] = pd.to_datetime(
        df[params["data"]["date_column"]], errors="coerce"
    )
    df = df.dropna(subset=[params["data"]["date_column"], "delay_reason"])
    df = df[(df["is_significant_delay"] == 1) & (df["delay_reason"] != "none")].copy()
    if df.empty:
        raise ValueError("No delayed flights with known delay_reason were found.")

    categorical_features = params["features"]["categorical"]
    numeric_features = params["features"]["numeric"]
    feature_columns = categorical_features + numeric_features

    train_df, val_df, test_df = time_based_split(
        df,
        params["data"]["date_column"],
        float(params["split"]["train_size"]),
        float(params["split"]["val_size"]),
    )

    model = build_model(categorical_features, numeric_features, settings)
    model.fit(train_df[feature_columns], train_df["delay_reason"])

    test_pred = model.predict(test_df[feature_columns])
    test_metrics = multiclass_classification_metrics(test_df["delay_reason"], pd.Series(test_pred))

    metrics = {
        "task_type": "delay_reason_classification",
        "model_type": settings["model_type"],
        "primary_metric": "macro_f1",
        "dataset_version": file_hash(features_path),
        "feature_set_version": file_hash(features_path),
        "split_strategy": params["split"]["strategy"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "classes": sorted(df["delay_reason"].unique().tolist()),
        "confusion_matrix": test_metrics["confusion_matrix"],
        "classification_report": test_metrics["classification_report"],
    }

    save_metrics(metrics, metrics_path)
    save_confusion_matrix(
        test_df["delay_reason"],
        pd.Series(test_pred),
        confusion_matrix_path,
        "Delay reason classification",
    )

    classifier = model.named_steps["classifier"]
    save_feature_importance(
        model_feature_names(model),
        classifier.feature_importances_.tolist(),
        feature_importance_csv,
        feature_importance_png,
        "Feature importance: delay reason",
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    dump(model, model_dir / "model.pkl")
    metadata = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "task_type": "delay_reason_classification",
        "model_type": settings["model_type"],
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "dataset_version": metrics["dataset_version"],
        "classes": metrics["classes"],
        "metrics": {
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        },
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    run_id = log_to_mlflow(
        run_name="train_reason_model",
        model=model,
        params={
            "reason_model": settings,
            "split": params["split"],
        },
        metrics={key: metrics[key] for key in ["macro_f1", "weighted_f1"]},
        artifacts=[metrics_path, confusion_matrix_path, feature_importance_csv, feature_importance_png],
        tags={
            "task_type": "delay_reason_classification",
            "dataset_version": metrics["dataset_version"],
            "feature_set_version": metrics["feature_set_version"],
        },
    )

    return {
        "task_type": "delay_reason_classification",
        "run_id": run_id,
        "metrics": metrics,
        "model_path": str(model_dir / "model.pkl"),
    }


def main() -> None:
    result = train_reason_model()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
