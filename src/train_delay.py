import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from src.evaluate import (
        binary_classification_metrics,
        save_confusion_matrix,
        save_feature_importance,
        save_metrics,
    )
except ModuleNotFoundError:
    from evaluate import (
        binary_classification_metrics,
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


def choose_threshold(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.10, 0.91, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_threshold = float(round(threshold, 2))
            best_f1 = float(score)
    return best_threshold, best_f1


def positive_class_probability(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(frame)
    classes = list(model.classes_)
    positive_index = classes.index(1) if 1 in classes else len(classes) - 1
    return probabilities[:, positive_index]


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


def train_delay_model(
    params_path: str | Path = "params.yaml",
    model_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = load_params(params_path)
    settings = dict(params["delay_model"])
    if model_overrides:
        settings.update(model_overrides)
    settings["random_state"] = params["split"]["random_state"]

    features_path = Path(params["data"]["features_path"])
    model_dir = Path("models/delay_model")
    metrics_path = Path("reports/metrics/delay_metrics.json")
    confusion_matrix_path = Path("reports/figures/confusion_matrix_delay.png")
    feature_importance_csv = Path("reports/metrics/feature_importance_delay.csv")
    feature_importance_png = Path("reports/figures/feature_importance_delay.png")

    df = pd.read_csv(features_path, encoding="utf-8-sig")
    df[params["data"]["date_column"]] = pd.to_datetime(
        df[params["data"]["date_column"]], errors="coerce"
    )
    df = df.dropna(subset=[params["data"]["date_column"], "is_significant_delay"])

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
    model.fit(train_df[feature_columns], train_df["is_significant_delay"])

    val_proba = positive_class_probability(model, val_df[feature_columns])
    threshold, validation_f1 = choose_threshold(val_df["is_significant_delay"], val_proba)

    test_proba = positive_class_probability(model, test_df[feature_columns])
    test_pred = (test_proba >= threshold).astype(int)
    test_metrics = binary_classification_metrics(
        test_df["is_significant_delay"],
        pd.Series(test_pred),
        pd.Series(test_proba),
    )

    metrics = {
        "task_type": "significant_delay_prediction",
        "model_type": settings["model_type"],
        "primary_metric": "f1",
        "dataset_version": file_hash(features_path),
        "feature_set_version": file_hash(features_path),
        "split_strategy": params["split"]["strategy"],
        "delay_threshold_minutes": params["target"]["delay_threshold_minutes"],
        "selected_threshold": threshold,
        "validation_f1": validation_f1,
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "roc_auc": test_metrics["roc_auc"],
        "pr_auc": test_metrics["pr_auc"],
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "confusion_matrix": test_metrics["confusion_matrix"],
    }

    save_metrics(metrics, metrics_path)
    save_confusion_matrix(
        test_df["is_significant_delay"],
        pd.Series(test_pred),
        confusion_matrix_path,
        "Significant delay prediction",
    )

    classifier = model.named_steps["classifier"]
    save_feature_importance(
        model_feature_names(model),
        classifier.feature_importances_.tolist(),
        feature_importance_csv,
        feature_importance_png,
        "Feature importance: significant delay",
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    dump(model, model_dir / "model.pkl")
    metadata = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "task_type": "significant_delay_prediction",
        "model_type": settings["model_type"],
        "threshold": threshold,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "dataset_version": metrics["dataset_version"],
        "metrics": {
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
        },
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    run_id = log_to_mlflow(
        run_name="train_delay_model",
        model=model,
        params={
            "delay_model": settings,
            "split": params["split"],
            "target": params["target"],
        },
        metrics={key: metrics[key] for key in ["precision", "recall", "f1", "roc_auc", "pr_auc"]},
        artifacts=[metrics_path, confusion_matrix_path, feature_importance_csv, feature_importance_png],
        tags={
            "task_type": "significant_delay_prediction",
            "dataset_version": metrics["dataset_version"],
            "feature_set_version": metrics["feature_set_version"],
        },
    )

    return {
        "task_type": "significant_delay_prediction",
        "run_id": run_id,
        "metrics": metrics,
        "model_path": str(model_dir / "model.pkl"),
    }


def main() -> None:
    result = train_delay_model()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
