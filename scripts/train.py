#!/usr/bin/env python3
"""
Eğitim pipeline'ı: dataset yükleme, öncü model, ana model eğitimi.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phishing_ai.config import (
    HYBRID_MODEL_PATH,
    HYBRID_MODEL_V1_PATH,
    HYBRID_MODEL_V2_PATH,
    HYBRID_MODEL_V3_PATH,
    MAIN_MODEL_PATH,
    MAIN_MODEL_V1_PATH,
    MAIN_MODEL_V2_PATH,
    MAIN_MODEL_V3_PATH,
    MODEL_METRICS_PATH,
    TEST_SAMPLES_PATH,
    TFIDF_V2_PARAM_GRID,
    V2_MIN_PRECISION,
    V2_RISK_WEIGHT_GRID,
)
from phishing_ai.data import get_train_test_split, load_combined_dataset, load_dataset
from phishing_ai.models import (
    compare_models,
    compare_models_with_embeddings,
    evaluate_probability_metrics,
    save_model,
    train_optimized_model,
    train_embedding_hybrid_model,
    train_main_model,
)


def _load_existing_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _persist_metrics(metrics_path: str | Path, model_name: str, metrics: dict) -> None:
    payload = _load_existing_json(metrics_path)
    payload[model_name] = metrics
    _write_json(metrics_path, payload)


def _persist_test_samples(samples_path: str | Path, texts, labels, limit: int = 200) -> None:
    samples = [
        {"text": text, "label": label}
        for text, label in list(zip(texts, labels))[:limit]
    ]
    _write_json(
        samples_path,
        {
            "sample_count": len(samples),
            "samples": samples,
        },
    )


def _archive_model_artifact(source_path: str | Path, archive_path: str | Path) -> None:
    source = Path(source_path)
    archive = Path(archive_path)
    if not source.exists() or archive.exists():
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, archive)


def archive_existing_models() -> None:
    _archive_model_artifact(MAIN_MODEL_PATH, MAIN_MODEL_V1_PATH)
    _archive_model_artifact(HYBRID_MODEL_PATH, HYBRID_MODEL_V1_PATH)


def _build_validation_split(texts, labels, random_state: int = 42):
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    if not label_counts or min(label_counts.values()) < 3 or len(texts) < 8:
        return list(texts), list(texts), list(labels), list(labels)
    stratify = labels if len(label_counts) > 1 and min(label_counts.values()) >= 2 else None
    X_subtrain, X_validation, y_subtrain, y_validation = train_test_split(
        list(texts),
        list(labels),
        test_size=0.25,
        random_state=random_state,
        stratify=stratify,
    )
    return X_subtrain, X_validation, y_subtrain, y_validation


def run_training_pipeline(
    dataset_path: str | Path = "data/raw/CEAS_08.csv",
    model_path: str | Path = MAIN_MODEL_PATH,
    metrics_path: str | Path = MODEL_METRICS_PATH,
    samples_path: str | Path = TEST_SAMPLES_PATH,
) -> dict:
    """Train and persist the phase-1 phishing model."""
    dataset = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = get_train_test_split(dataset)
    model_metrics = compare_models(X_train, y_train, X_test, y_test)

    final_model = train_main_model(dataset["text"].tolist(), dataset["label"].tolist())
    save_model(final_model, model_path)
    _persist_metrics(metrics_path, "main", model_metrics)
    _persist_test_samples(samples_path, X_test.tolist(), y_test.tolist())

    return {
        "dataset_rows": len(dataset),
        "model_metrics": model_metrics,
        "saved_model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "samples_path": str(samples_path),
    }


def run_phase2_hybrid_pipeline(
    dataset_path: str | Path = "data/raw/CEAS_08.csv",
    model_path: str | Path = HYBRID_MODEL_PATH,
    embedding_model=None,
    max_rows: int | None = None,
    random_state: int = 42,
    metrics_path: str | Path = MODEL_METRICS_PATH,
    samples_path: str | Path = TEST_SAMPLES_PATH,
) -> dict:
    """Train and persist the optional phase-2 hybrid (TF-IDF + security + embeddings) model."""
    dataset = load_dataset(dataset_path)
    if max_rows is not None and len(dataset) > max_rows:
        dataset = dataset.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    X_train, X_test, y_train, y_test = get_train_test_split(dataset)
    hybrid_metrics = compare_models_with_embeddings(
        X_train.tolist(),
        y_train.tolist(),
        X_test.tolist(),
        y_test.tolist(),
        embedding_model=embedding_model,
    )
    hybrid_model = train_embedding_hybrid_model(
        dataset["text"].tolist(),
        dataset["label"].tolist(),
        embedding_model=embedding_model,
    )
    save_model(hybrid_model, model_path)
    _persist_metrics(metrics_path, "hybrid", hybrid_metrics)
    if not Path(samples_path).exists():
        _persist_test_samples(samples_path, X_test.tolist(), y_test.tolist())
    return {
        "dataset_rows": len(dataset),
        "hybrid_metrics": hybrid_metrics,
        "saved_hybrid_model_path": str(model_path),
        "metrics_path": str(metrics_path),
    }


def _select_best_v2_candidate(
    X_train,
    y_train,
    *,
    use_embeddings: bool,
    embedding_model=None,
    verbose: bool = True,
):
    X_subtrain, X_validation, y_subtrain, y_validation = _build_validation_split(X_train, y_train)
    best_metrics = None
    best_vectorizer_params = None
    total = len(TFIDF_V2_PARAM_GRID)

    for idx, vectorizer_params in enumerate(TFIDF_V2_PARAM_GRID, 1):
        if verbose:
            print(f"  [Grid {idx}/{total}] TF-IDF min_df={vectorizer_params.get('min_df', '?')} ...", flush=True)
        try:
            _, metrics = train_optimized_model(
                X_subtrain,
                y_subtrain,
                X_validation,
                y_validation,
                vectorizer_params=vectorizer_params,
                risk_weight_grid=V2_RISK_WEIGHT_GRID,
                use_embeddings=use_embeddings,
                embedding_model=embedding_model,
                min_precision=V2_MIN_PRECISION,
                model_version="v2",
            )
        except ValueError:
            if verbose:
                print(f"    -> Skipped (ValueError)", flush=True)
            continue
        if best_metrics is None:
            best_metrics = metrics
            best_vectorizer_params = vectorizer_params
            continue
        if metrics["recall"] > best_metrics["recall"]:
            best_metrics = metrics
            best_vectorizer_params = vectorizer_params
            continue
        if metrics["recall"] == best_metrics["recall"] and metrics["brier_score"] < best_metrics["brier_score"]:
            best_metrics = metrics
            best_vectorizer_params = vectorizer_params

    return best_vectorizer_params, best_metrics


def _evaluate_saved_model(model, X_test, y_test, embedding_model=None) -> dict:
    from phishing_ai.models import predict

    probabilities = predict(model, X_test, embedding_model=embedding_model)["probabilities"]
    return evaluate_probability_metrics(y_test, probabilities, float(model.get("threshold", 0.5)))


def run_v2_training_pipeline(
    dataset_path: str | Path = "data/raw/CEAS_08.csv",
    model_path: str | Path = MAIN_MODEL_V2_PATH,
    max_rows: int | None = None,
    random_state: int = 42,
    metrics_path: str | Path = MODEL_METRICS_PATH,
    samples_path: str | Path = TEST_SAMPLES_PATH,
) -> dict:
    """Train and persist the recall-oriented calibrated phase-1 v2 model."""
    archive_existing_models()
    dataset = load_dataset(dataset_path)
    if max_rows is not None and len(dataset) > max_rows:
        dataset = dataset.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    X_train, X_test, y_train, y_test = get_train_test_split(dataset)
    best_vectorizer_params, validation_metrics = _select_best_v2_candidate(
        X_train.tolist(),
        y_train.tolist(),
        use_embeddings=False,
    )
    if best_vectorizer_params is None or validation_metrics is None:
        raise RuntimeError("Failed to produce a v2 main model candidate.")
    best_model, _ = train_optimized_model(
        X_train.tolist(),
        y_train.tolist(),
        X_train.tolist(),
        y_train.tolist(),
        vectorizer_params=best_vectorizer_params,
        risk_weight_grid=V2_RISK_WEIGHT_GRID,
        use_embeddings=False,
        min_precision=V2_MIN_PRECISION,
        model_version="v2",
    )
    best_model["threshold"] = validation_metrics["selected_threshold"]
    best_model["risk_weights"] = validation_metrics["risk_weights"]
    save_model(best_model, model_path)
    test_metrics = _evaluate_saved_model(best_model, X_test.tolist(), y_test.tolist())
    metrics_payload = {**test_metrics, "validation": validation_metrics}
    _persist_metrics(metrics_path, "main_v2", metrics_payload)
    _persist_test_samples(samples_path, X_test.tolist(), y_test.tolist())
    return {
        "dataset_rows": len(dataset),
        "model_metrics": metrics_payload,
        "saved_model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "samples_path": str(samples_path),
    }


def run_v3_training_pipeline(
    dataset_path: str | Path = "data/raw/CEAS_08.csv",
    model_path: str | Path = MAIN_MODEL_V3_PATH,
    hf_max_rows: int | None = 5000,
    max_rows_total: int | None = None,
    random_state: int = 42,
    metrics_path: str | Path = MODEL_METRICS_PATH,
    samples_path: str | Path = TEST_SAMPLES_PATH,
) -> dict:
    """Train and persist the v3 main model on combined CEAS + HF dataset."""
    print("Loading combined CEAS + HF dataset...", flush=True)
    dataset = load_combined_dataset(
        ceas_path=dataset_path,
        hf_max_rows=hf_max_rows,
        max_rows_total=max_rows_total,
        random_state=random_state,
    )
    print(f"Dataset loaded: {len(dataset):,} rows (CEAS + HF, deduplicated)", flush=True)
    X_train, X_test, y_train, y_test = get_train_test_split(dataset, random_state=random_state)
    print("main_v3: Grid search (CPU, TF-IDF + LR)...", flush=True)
    best_vectorizer_params, validation_metrics = _select_best_v2_candidate(
        X_train.tolist(),
        y_train.tolist(),
        use_embeddings=False,
    )
    if best_vectorizer_params is None or validation_metrics is None:
        raise RuntimeError("Failed to produce a v3 main model candidate.")
    best_model, _ = train_optimized_model(
        X_train.tolist(),
        y_train.tolist(),
        X_train.tolist(),
        y_train.tolist(),
        vectorizer_params=best_vectorizer_params,
        risk_weight_grid=V2_RISK_WEIGHT_GRID,
        use_embeddings=False,
        min_precision=V2_MIN_PRECISION,
        model_version="v2",
    )
    best_model["threshold"] = validation_metrics["selected_threshold"]
    best_model["risk_weights"] = validation_metrics["risk_weights"]
    save_model(best_model, model_path)
    test_metrics = _evaluate_saved_model(best_model, X_test.tolist(), y_test.tolist())
    metrics_payload = {**test_metrics, "validation": validation_metrics}
    _persist_metrics(metrics_path, "main_v3", metrics_payload)
    _persist_test_samples(samples_path, X_test.tolist(), y_test.tolist())
    return {
        "dataset_rows": len(dataset),
        "model_metrics": metrics_payload,
        "saved_model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "samples_path": str(samples_path),
    }


def run_phase2_hybrid_v3_pipeline(
    dataset_path: str | Path = "data/raw/CEAS_08.csv",
    model_path: str | Path = HYBRID_MODEL_V3_PATH,
    embedding_model=None,
    hf_max_rows: int | None = 5000,
    max_rows_total: int | None = None,
    random_state: int = 42,
    metrics_path: str | Path = MODEL_METRICS_PATH,
    samples_path: str | Path = TEST_SAMPLES_PATH,
) -> dict:
    """Train and persist the v3 hybrid model on combined CEAS + HF dataset."""
    print("Loading combined CEAS + HF dataset for hybrid_v3...", flush=True)
    dataset = load_combined_dataset(
        ceas_path=dataset_path,
        hf_max_rows=hf_max_rows,
        max_rows_total=max_rows_total,
        random_state=random_state,
    )
    print(f"Dataset loaded: {len(dataset):,} rows", flush=True)
    X_train, X_test, y_train, y_test = get_train_test_split(dataset, random_state=random_state)
    print("hybrid_v3: Grid search (embeddings may use GPU)...", flush=True)
    best_vectorizer_params, validation_metrics = _select_best_v2_candidate(
        X_train.tolist(),
        y_train.tolist(),
        use_embeddings=True,
        embedding_model=embedding_model,
    )
    if best_vectorizer_params is None or validation_metrics is None:
        raise RuntimeError("Failed to produce a v3 hybrid model candidate.")
    best_model, _ = train_optimized_model(
        X_train.tolist(),
        y_train.tolist(),
        X_train.tolist(),
        y_train.tolist(),
        vectorizer_params=best_vectorizer_params,
        risk_weight_grid=V2_RISK_WEIGHT_GRID,
        use_embeddings=True,
        embedding_model=embedding_model,
        min_precision=V2_MIN_PRECISION,
        model_version="v2",
    )
    best_model["threshold"] = validation_metrics["selected_threshold"]
    best_model["risk_weights"] = validation_metrics["risk_weights"]
    save_model(best_model, model_path)
    test_metrics = _evaluate_saved_model(
        best_model,
        X_test.tolist(),
        y_test.tolist(),
        embedding_model=embedding_model,
    )
    metrics_payload = {**test_metrics, "validation": validation_metrics}
    _persist_metrics(metrics_path, "hybrid_v3", metrics_payload)
    if not Path(samples_path).exists():
        _persist_test_samples(samples_path, X_test.tolist(), y_test.tolist())
    return {
        "dataset_rows": len(dataset),
        "hybrid_metrics": metrics_payload,
        "saved_hybrid_model_path": str(model_path),
        "metrics_path": str(metrics_path),
    }


def run_phase2_hybrid_v2_pipeline(
    dataset_path: str | Path = "data/raw/CEAS_08.csv",
    model_path: str | Path = HYBRID_MODEL_V2_PATH,
    embedding_model=None,
    max_rows: int | None = None,
    random_state: int = 42,
    metrics_path: str | Path = MODEL_METRICS_PATH,
    samples_path: str | Path = TEST_SAMPLES_PATH,
) -> dict:
    """Train and persist the calibrated hybrid v2 model."""
    archive_existing_models()
    dataset = load_dataset(dataset_path)
    if max_rows is not None and len(dataset) > max_rows:
        dataset = dataset.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    X_train, X_test, y_train, y_test = get_train_test_split(dataset)
    best_vectorizer_params, validation_metrics = _select_best_v2_candidate(
        X_train.tolist(),
        y_train.tolist(),
        use_embeddings=True,
        embedding_model=embedding_model,
    )
    if best_vectorizer_params is None or validation_metrics is None:
        raise RuntimeError("Failed to produce a v2 hybrid model candidate.")
    best_model, _ = train_optimized_model(
        X_train.tolist(),
        y_train.tolist(),
        X_train.tolist(),
        y_train.tolist(),
        vectorizer_params=best_vectorizer_params,
        risk_weight_grid=V2_RISK_WEIGHT_GRID,
        use_embeddings=True,
        embedding_model=embedding_model,
        min_precision=V2_MIN_PRECISION,
        model_version="v2",
    )
    best_model["threshold"] = validation_metrics["selected_threshold"]
    best_model["risk_weights"] = validation_metrics["risk_weights"]
    save_model(best_model, model_path)
    test_metrics = _evaluate_saved_model(
        best_model,
        X_test.tolist(),
        y_test.tolist(),
        embedding_model=embedding_model,
    )
    metrics_payload = {**test_metrics, "validation": validation_metrics}
    _persist_metrics(metrics_path, "hybrid_v2", metrics_payload)
    if not Path(samples_path).exists():
        _persist_test_samples(samples_path, X_test.tolist(), y_test.tolist())
    return {
        "dataset_rows": len(dataset),
        "hybrid_metrics": metrics_payload,
        "saved_hybrid_model_path": str(model_path),
        "metrics_path": str(metrics_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train phishing email models.")
    parser.add_argument(
        "--dataset-path",
        default="data/raw/CEAS_08.csv",
        help="Path to CEAS_08.csv",
    )
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Also train the optional hybrid embedding model",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Also train the calibrated recall-first v2 models",
    )
    parser.add_argument(
        "--v2-only",
        action="store_true",
        help="Skip legacy training and only run the v2 pipelines",
    )
    parser.add_argument(
        "--phase2-only",
        action="store_true",
        help="Skip phase-1 training and only run phase-2 (requires dataset and will overwrite hybrid model)",
    )
    parser.add_argument(
        "--phase2-max-rows",
        type=int,
        default=8000,
        help="Max rows to use for phase-2 hybrid training (default: 8000 for speed)",
    )
    parser.add_argument(
        "--v2-max-rows",
        type=int,
        default=20000,
        help="Max rows to use for phase-1 v2 training (default: 20000 for speed)",
    )
    parser.add_argument(
        "--v3",
        action="store_true",
        help="Train v3 models on combined CEAS + HuggingFace dataset",
    )
    parser.add_argument(
        "--v3-only",
        action="store_true",
        help="Skip legacy/v2 training and only run v3 pipelines",
    )
    parser.add_argument(
        "--hf-max-rows",
        type=int,
        default=5000,
        help="Max rows from HuggingFace dataset for v3 (default: 5000)",
    )
    parser.add_argument(
        "--max-rows-total",
        type=int,
        default=None,
        help="Max total rows for v3 combined dataset (optional)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.v2 or args.v2_only:
        archive_existing_models()

    if args.v3 or args.v3_only:
        report_v3 = run_v3_training_pipeline(
            dataset_path=args.dataset_path,
            hf_max_rows=args.hf_max_rows,
            max_rows_total=args.max_rows_total,
        )
        print("Training report (phase 1 v3 - CEAS + HF combined):")
        print(report_v3["model_metrics"])
        print(f"Saved v3 model: {report_v3['saved_model_path']}")

        if args.phase2 or args.phase2_only:
            report_v3_hybrid = run_phase2_hybrid_v3_pipeline(
                dataset_path=args.dataset_path,
                hf_max_rows=args.hf_max_rows,
                max_rows_total=args.max_rows_total,
            )
            print("Training report (phase 2 v3 - hybrid embeddings):")
            print(report_v3_hybrid["hybrid_metrics"])
            print(f"Saved v3 hybrid model: {report_v3_hybrid['saved_hybrid_model_path']}")
    elif args.v2 or args.v2_only:
        report3 = run_v2_training_pipeline(
            dataset_path=args.dataset_path,
            max_rows=args.v2_max_rows,
        )
        print("Training report (phase 1 v2):")
        print(report3["model_metrics"])
        print(f"Saved v2 model: {report3['saved_model_path']}")

        if args.phase2 or args.phase2_only:
            report4 = run_phase2_hybrid_v2_pipeline(
                dataset_path=args.dataset_path,
                max_rows=args.phase2_max_rows,
            )
            print("Training report (phase 2 v2 - hybrid embeddings):")
            print(report4["hybrid_metrics"])
            print(f"Saved v2 hybrid model: {report4['saved_hybrid_model_path']}")
    else:
        if not args.phase2_only:
            report = run_training_pipeline(dataset_path=args.dataset_path)
            print("Training report (phase 1):")
            for model_name, metrics in report["model_metrics"].items():
                print(f"- {model_name}: {metrics}")
            print(f"Saved model: {report['saved_model_path']}")

        if args.phase2 or args.phase2_only:
            report2 = run_phase2_hybrid_pipeline(
                dataset_path=args.dataset_path,
                max_rows=args.phase2_max_rows,
            )
            print("Training report (phase 2 - hybrid embeddings):")
            for model_name, metrics in report2["hybrid_metrics"].items():
                print(f"- {model_name}: {metrics}")
            print(f"Saved hybrid model: {report2['saved_hybrid_model_path']}")

        if args.v2:
            report3 = run_v2_training_pipeline(
                dataset_path=args.dataset_path,
                max_rows=args.v2_max_rows,
            )
            print("Training report (phase 1 v2):")
            print(report3["model_metrics"])
            print(f"Saved v2 model: {report3['saved_model_path']}")

            if args.phase2 or args.phase2_only:
                report4 = run_phase2_hybrid_v2_pipeline(
                    dataset_path=args.dataset_path,
                    max_rows=args.phase2_max_rows,
                )
                print("Training report (phase 2 v2 - hybrid embeddings):")
                print(report4["hybrid_metrics"])
                print(f"Saved v2 hybrid model: {report4['saved_hybrid_model_path']}")
