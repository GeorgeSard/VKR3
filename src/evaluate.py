import json
import os
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = Path(".cache/matplotlib").resolve()
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def binary_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series,
) -> dict[str, Any]:
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": None,
        "pr_auc": None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    return metrics


def multiclass_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, Any]:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, default=_json_default)


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: str | Path,
    title: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    display.ax_.set_title(title)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def save_feature_importance(
    feature_names: list[str],
    importances: list[float],
    csv_path: str | Path,
    figure_path: str | Path,
    title: str,
    top_n: int = 20,
) -> None:
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    csv_output = Path(csv_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_output, index=False)

    figure_output = Path(figure_path)
    figure_output.parent.mkdir(parents=True, exist_ok=True)
    top = frame.head(top_n).sort_values("importance")

    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["importance"])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(figure_output)
    plt.close()
