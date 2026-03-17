#!/usr/bin/env python3
"""
Eğitim pipeline'ı: dataset yükleme, öncü model, ana model eğitimi.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phishing_ai.config import MAIN_MODEL_PATH
from phishing_ai.data import get_train_test_split, load_dataset
from phishing_ai.models import compare_models, save_model, train_main_model


def run_training_pipeline(
    dataset_path: str | Path = "data/raw/CEAS_08.csv",
    model_path: str | Path = MAIN_MODEL_PATH,
) -> dict:
    """Train and persist the phase-1 phishing model."""
    dataset = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = get_train_test_split(dataset)
    model_metrics = compare_models(X_train, y_train, X_test, y_test)

    final_model = train_main_model(dataset["text"].tolist(), dataset["label"].tolist())
    save_model(final_model, model_path)

    return {
        "dataset_rows": len(dataset),
        "model_metrics": model_metrics,
        "saved_model_path": str(model_path),
    }

if __name__ == "__main__":
    report = run_training_pipeline()
    print("Training report:")
    for model_name, metrics in report["model_metrics"].items():
        print(f"- {model_name}: {metrics}")
    print(f"Saved model: {report['saved_model_path']}")
