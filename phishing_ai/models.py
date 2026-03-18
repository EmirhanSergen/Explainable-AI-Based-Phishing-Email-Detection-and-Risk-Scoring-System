"""
Öncü model, ana model ve model karşılaştırması.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from phishing_ai.config import V2_MIN_PRECISION, W_PROB, W_URL, W_KW
from phishing_ai.features import (
    build_embedding_matrix,
    build_security_feature_matrix,
    get_embedding_model,
    get_security_feature_names,
    get_tfidf_vectorizer,
    normalize_text,
)
from phishing_ai.risk import compute_risk_score


def _make_logistic_regression(class_weight: str | dict | None = None) -> LogisticRegression:
    # SAGA is stable on large sparse matrices and reduces numerical warnings
    # compared to some default solvers on perfectly/semi-separable data.
    return LogisticRegression(
        max_iter=2000,
        solver="saga",
        penalty="l2",
        n_jobs=-1,
        random_state=42,
        class_weight=class_weight,
    )


def _prepare_texts(texts) -> list[str]:
    return [normalize_text(text) for text in texts]


def _build_feature_matrix(
    texts,
    vectorizer=None,
    fit=False,
    use_embeddings=False,
    embedding_model=None,
    vectorizer_params: dict | None = None,
):
    prepared_texts = _prepare_texts(texts)
    tfidf_vectorizer = vectorizer or get_tfidf_vectorizer(**(vectorizer_params or {}))
    if fit:
        tfidf_matrix = tfidf_vectorizer.fit_transform(prepared_texts)
    else:
        tfidf_matrix = tfidf_vectorizer.transform(prepared_texts)

    security_matrix = csr_matrix(build_security_feature_matrix(prepared_texts))
    matrices = [tfidf_matrix, security_matrix]
    fitted_embedding_model = embedding_model

    if use_embeddings:
        embedding_matrix, fitted_embedding_model = build_embedding_matrix(
            prepared_texts,
            embedding_model=embedding_model,
        )
        matrices.append(csr_matrix(embedding_matrix))

    full_matrix = hstack(matrices, format="csr")
    return full_matrix, tfidf_vectorizer, fitted_embedding_model


def _build_artifact(
    classifier,
    vectorizer,
    embedding_model_name: str | None = None,
    embedding_dims: int = 0,
    calibrated_classifier=None,
    threshold: float = 0.5,
    risk_weights: dict | None = None,
    risk_thresholds: dict | None = None,
    model_version: str = "v1",
) -> dict:
    tfidf_feature_names = vectorizer.get_feature_names_out().tolist()
    embedding_feature_names = [
        f"embedding_{index}" for index in range(embedding_dims)
    ]
    return {
        "classifier": classifier,
        "calibrated_classifier": calibrated_classifier,
        "vectorizer": vectorizer,
        # Persist only the name; the model object is heavy and not reliably pickleable.
        "embedding_model_name": embedding_model_name,
        "security_feature_names": get_security_feature_names(),
        "threshold": threshold,
        "risk_weights": risk_weights or {},
        "risk_thresholds": risk_thresholds or {},
        "model_version": model_version,
        "feature_names": tfidf_feature_names
        + get_security_feature_names()
        + embedding_feature_names,
    }


def _metric_dict(y_true, y_pred) -> dict:
    normalized_y_true = _normalize_label_sequence(y_true)
    normalized_y_pred = _normalize_label_sequence(y_pred)
    return {
        "accuracy": accuracy_score(normalized_y_true, normalized_y_pred),
        "precision": precision_score(
            normalized_y_true,
            normalized_y_pred,
            pos_label="phishing",
            zero_division=0,
        ),
        "recall": recall_score(
            normalized_y_true,
            normalized_y_pred,
            pos_label="phishing",
            zero_division=0,
        ),
        "f1": f1_score(
            normalized_y_true,
            normalized_y_pred,
            pos_label="phishing",
            zero_division=0,
        ),
    }


def _labels_from_probabilities(probabilities: list[float], threshold: float) -> list[str]:
    return ["phishing" if probability >= threshold else "legitimate" for probability in probabilities]


def _normalize_label_value(value) -> str:
    """Collapse array-like label wrappers into a single scalar string label."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalize_label_value(value.item())
        flattened = value.reshape(-1).tolist()
        if len(flattened) != 1:
            raise ValueError(f"Expected scalar label, received array-like label: {value!r}")
        return _normalize_label_value(flattened[0])
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected scalar label, received nested label: {value!r}")
        return _normalize_label_value(value[0])
    return str(value)


def _normalize_label_sequence(labels) -> list[str]:
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    return [_normalize_label_value(label) for label in labels]


def _normalize_probability_sequence(probabilities) -> list[float]:
    if hasattr(probabilities, "tolist"):
        probabilities = probabilities.tolist()
    return [float(probability) for probability in probabilities]


def select_probability_threshold(
    y_true,
    probabilities,
    min_precision: float = V2_MIN_PRECISION,
) -> float:
    """Choose the highest-recall threshold that satisfies a minimum precision floor."""
    normalized_y_true = _normalize_label_sequence(y_true)
    normalized_probabilities = _normalize_probability_sequence(probabilities)
    if not normalized_probabilities:
        return 0.5

    best_threshold = 0.5
    best_recall = -1.0
    best_precision = -1.0
    candidates = sorted(set(normalized_probabilities), reverse=True)
    candidates.append(1.0)

    for threshold in candidates:
        predictions = _labels_from_probabilities(normalized_probabilities, threshold)
        precision = precision_score(
            normalized_y_true,
            predictions,
            pos_label="phishing",
            zero_division=0,
        )
        recall = recall_score(
            normalized_y_true,
            predictions,
            pos_label="phishing",
            zero_division=0,
        )
        if precision < min_precision:
            continue
        if recall > best_recall or (recall == best_recall and precision > best_precision):
            best_threshold = threshold
            best_recall = recall
            best_precision = precision

    if best_recall >= 0:
        return float(best_threshold)

    # Fall back to the best F1 threshold if no candidate can satisfy the precision floor.
    fallback_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        predictions = _labels_from_probabilities(normalized_probabilities, threshold)
        score = f1_score(
            normalized_y_true,
            predictions,
            pos_label="phishing",
            zero_division=0,
        )
        if score > best_f1:
            best_f1 = score
            fallback_threshold = threshold
    return float(fallback_threshold)


def evaluate_probability_metrics(y_true, probabilities, threshold: float) -> dict:
    normalized_y_true = _normalize_label_sequence(y_true)
    normalized_probabilities = _normalize_probability_sequence(probabilities)
    predictions = _labels_from_probabilities(normalized_probabilities, threshold)
    phishing_total = sum(1 for label in normalized_y_true if label == "phishing")
    legitimate_total = sum(1 for label in normalized_y_true if label == "legitimate")
    false_negatives = sum(
        1
        for actual, predicted in zip(normalized_y_true, predictions)
        if actual == "phishing" and predicted != "phishing"
    )
    false_positives = sum(
        1
        for actual, predicted in zip(normalized_y_true, predictions)
        if actual == "legitimate" and predicted == "phishing"
    )
    metrics = _metric_dict(normalized_y_true, predictions)
    metrics.update(
        {
            "average_precision": average_precision_score(
                [1 if label == "phishing" else 0 for label in normalized_y_true],
                normalized_probabilities,
            ),
            "brier_score": brier_score_loss(
                [1 if label == "phishing" else 0 for label in normalized_y_true],
                normalized_probabilities,
            ),
            "selected_threshold": float(threshold),
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "phishing_support": phishing_total,
            "legitimate_support": legitimate_total,
        }
    )
    return metrics


def train_pioneer_model(X_train, y_train):
    """
    Öncü model: TF-IDF + security features ile Logistic Regression.
    p_phishing_baseline üretir.
    """
    X_matrix, vectorizer, _ = _build_feature_matrix(X_train, fit=True)
    classifier = _make_logistic_regression()
    classifier.fit(X_matrix, y_train)
    return _build_artifact(classifier, vectorizer)


def train_main_model(X_train, y_train):
    """
    Ana model: TF-IDF + security features + p_phishing_baseline ile nihai classifier.
    """
    return train_pioneer_model(X_train, y_train)


def compare_models(X_train, y_train, X_test, y_test):
    """Compare the phase-1 baseline models on a shared feature space."""
    X_train_matrix, vectorizer, _ = _build_feature_matrix(X_train, fit=True)
    X_test_matrix, _, _ = _build_feature_matrix(X_test, vectorizer=vectorizer, fit=False)

    models = {
        "logistic_regression": _make_logistic_regression(),
        "linear_svm": LinearSVC(random_state=42),
        "naive_bayes": MultinomialNB(),
    }
    metrics = {}
    for name, model in models.items():
        model.fit(X_train_matrix, y_train)
        predictions = model.predict(X_test_matrix)
        metrics[name] = _metric_dict(y_test, predictions)
    return metrics


def train_embedding_hybrid_model(X_train, y_train, embedding_model=None):
    """Train the optional phase-2 hybrid model with TF-IDF, security, and embeddings."""
    X_matrix, vectorizer, fitted_embedding_model = _build_feature_matrix(
        X_train,
        fit=True,
        use_embeddings=True,
        embedding_model=embedding_model,
    )
    classifier = _make_logistic_regression()
    classifier.fit(X_matrix, y_train)
    embedding_dims = getattr(X_matrix, "shape", (0, 0))[1] - (
        len(vectorizer.get_feature_names_out()) + len(get_security_feature_names())
    )
    embedding_model_name = None
    if fitted_embedding_model is not None:
        embedding_model_name = getattr(fitted_embedding_model, "model_name", None) or getattr(
            fitted_embedding_model, "name_or_path", None
        )
    return _build_artifact(
        classifier,
        vectorizer,
        embedding_model_name=embedding_model_name,
        embedding_dims=max(embedding_dims, 0),
    )


def compare_models_with_embeddings(X_train, y_train, X_test, y_test, embedding_model=None):
    """Evaluate a hybrid model that augments phase-1 features with embeddings."""
    X_train_matrix, vectorizer, fitted_embedding_model = _build_feature_matrix(
        X_train,
        fit=True,
        use_embeddings=True,
        embedding_model=embedding_model,
    )
    X_test_matrix, _, _ = _build_feature_matrix(
        X_test,
        vectorizer=vectorizer,
        fit=False,
        use_embeddings=True,
        embedding_model=fitted_embedding_model,
    )
    classifier = _make_logistic_regression()
    classifier.fit(X_train_matrix, y_train)
    predictions = classifier.predict(X_test_matrix)
    return {"logistic_regression_hybrid": _metric_dict(y_test, predictions)}


def _select_risk_weights(probabilities, texts, y_true, risk_weight_grid) -> dict:
    best_weights = risk_weight_grid[0]
    best_score = float("-inf")
    security_matrix = build_security_feature_matrix(_prepare_texts(texts))

    for candidate in risk_weight_grid:
        risk_scores = [
            compute_risk_score(
                p_phishing_final=float(probability),
                url_count=int(row[0]),
                keyword_count=int(row[2]),
                w_prob=candidate["w_prob"],
                w_url=candidate["w_url"],
                w_kw=candidate["w_kw"],
            )
            for probability, row in zip(probabilities, security_matrix)
        ]
        phishing_high = sum(
            1
            for label, score in zip(y_true, risk_scores)
            if label == "phishing" and score >= 51
        )
        legitimate_low = sum(
            1
            for label, score in zip(y_true, risk_scores)
            if label == "legitimate" and score <= 25
        )
        candidate_score = (2 * phishing_high) + legitimate_low
        if candidate_score > best_score:
            best_score = candidate_score
            best_weights = candidate
    return best_weights


def calibrate_risk_thresholds(
    probabilities: list[float],
    texts: list[str],
    labels: list[str],
    risk_weights: dict,
) -> dict:
    """Find optimal Low/Medium/High/Critical risk score boundaries from data.

    Grid-searches (t1, t2, t3) triplets and maximises:
        2 * (phishing emails with score >= t2) + (legitimate emails with score < t1)

    Returns a dict with keys: low_medium, medium_high, high_critical.
    """
    w_prob = risk_weights.get("w_prob", W_PROB)
    w_url = risk_weights.get("w_url", W_URL)
    w_kw = risk_weights.get("w_kw", W_KW)

    security_matrix = build_security_feature_matrix(_prepare_texts(texts))
    risk_scores = [
        compute_risk_score(
            p_phishing_final=float(p),
            url_count=int(row[0]),
            keyword_count=int(row[2]),
            w_prob=w_prob,
            w_url=w_url,
            w_kw=w_kw,
        )
        for p, row in zip(probabilities, security_matrix)
    ]

    best_score = -1
    best_thresholds = {"low_medium": 25, "medium_high": 50, "high_critical": 75}

    for t1 in range(10, 45, 5):
        for t2 in range(t1 + 10, 70, 5):
            for t3 in range(t2 + 10, 95, 5):
                phishing_high = sum(
                    1 for label, s in zip(labels, risk_scores)
                    if label == "phishing" and s >= t2
                )
                legitimate_low = sum(
                    1 for label, s in zip(labels, risk_scores)
                    if label == "legitimate" and s < t1
                )
                candidate = (2 * phishing_high) + legitimate_low
                if candidate > best_score:
                    best_score = candidate
                    best_thresholds = {
                        "low_medium": t1,
                        "medium_high": t2,
                        "high_critical": t3,
                    }

    return best_thresholds


def train_optimized_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
    *,
    vectorizer_params: dict | None = None,
    risk_weight_grid,
    use_embeddings: bool = False,
    embedding_model=None,
    min_precision: float = V2_MIN_PRECISION,
    calibration_method: str = "sigmoid",
    model_version: str = "v2",
) -> tuple[dict, dict]:
    X_train_matrix, vectorizer, fitted_embedding_model = _build_feature_matrix(
        X_train,
        fit=True,
        use_embeddings=use_embeddings,
        embedding_model=embedding_model,
        vectorizer_params=vectorizer_params,
    )
    X_validation_matrix, _, _ = _build_feature_matrix(
        X_validation,
        vectorizer=vectorizer,
        fit=False,
        use_embeddings=use_embeddings,
        embedding_model=fitted_embedding_model,
    )

    classifier = _make_logistic_regression(class_weight="balanced")
    classifier.fit(X_train_matrix, y_train)

    label_counts = {}
    for label in y_train:
        label_counts[label] = label_counts.get(label, 0) + 1
    min_class_count = min(label_counts.values())
    calibrated_classifier = None
    probability_model = classifier
    effective_calibration_method = "none"
    if min_class_count >= 2:
        calibration_cv = max(2, min(3, min_class_count))
        calibrated_classifier = CalibratedClassifierCV(
            estimator=_make_logistic_regression(class_weight="balanced"),
            method=calibration_method,
            cv=calibration_cv,
        )
        calibrated_classifier.fit(X_train_matrix, y_train)
        probability_model = calibrated_classifier
        effective_calibration_method = calibration_method

    phishing_index = list(probability_model.classes_).index("phishing")
    probabilities = probability_model.predict_proba(X_validation_matrix)[:, phishing_index].tolist()
    threshold = select_probability_threshold(
        y_validation,
        probabilities,
        min_precision=min_precision,
    )
    risk_weights = _select_risk_weights(probabilities, X_validation, y_validation, risk_weight_grid)
    risk_thresholds = calibrate_risk_thresholds(
        probabilities, X_validation, y_validation, risk_weights
    )
    metrics = evaluate_probability_metrics(y_validation, probabilities, threshold)
    metrics.update(
        {
            "calibration_method": effective_calibration_method,
            "vectorizer_params": vectorizer_params or {},
            "risk_weights": risk_weights,
            "risk_thresholds": risk_thresholds,
            "use_embeddings": use_embeddings,
        }
    )

    embedding_dims = 0
    if use_embeddings:
        embedding_dims = getattr(X_train_matrix, "shape", (0, 0))[1] - (
            len(vectorizer.get_feature_names_out()) + len(get_security_feature_names())
        )
    embedding_model_name = None
    if fitted_embedding_model is not None:
        embedding_model_name = getattr(fitted_embedding_model, "model_name", None) or getattr(
            fitted_embedding_model, "name_or_path", None
        )

    artifact = _build_artifact(
        classifier,
        vectorizer,
        embedding_model_name=embedding_model_name,
        embedding_dims=max(embedding_dims, 0),
        calibrated_classifier=calibrated_classifier,
        threshold=threshold,
        risk_weights=risk_weights,
        risk_thresholds=risk_thresholds,
        model_version=model_version,
    )
    return artifact, metrics


def save_model(model, path: str):
    """Model kaydet."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    """Model yükle."""
    return joblib.load(path)


def _get_feature_group_slices(model) -> dict:
    vectorizer = model.get("vectorizer")
    if vectorizer is None:
        return {"tfidf": slice(0, 0), "security": slice(0, 0), "embedding": slice(0, 0)}

    tfidf_len = len(vectorizer.get_feature_names_out())
    security_len = len(get_security_feature_names())
    base_len = tfidf_len + security_len

    expected_len = getattr(model.get("classifier"), "n_features_in_", base_len)
    embedding_len = max(int(expected_len - base_len), 0)

    return {
        "tfidf": slice(0, tfidf_len),
        "security": slice(tfidf_len, tfidf_len + security_len),
        "embedding": slice(tfidf_len + security_len, tfidf_len + security_len + embedding_len),
    }


def _try_get_group_contributions(model, X_matrix) -> dict | None:
    """
    Return per-group logit contributions for binary LogisticRegression.
    Uses linear decomposition: logit = bias + sum_i w_i * x_i
    """
    clf = model.get("classifier")
    if clf is None or not hasattr(clf, "coef_"):
        return None

    coef = getattr(clf, "coef_", None)
    intercept = getattr(clf, "intercept_", None)
    if coef is None or intercept is None:
        return None

    coef = np.asarray(coef)
    if coef.ndim != 2 or coef.shape[0] != 1:
        # Only support binary logistic regression for now.
        return None

    w = coef[0]
    slices = _get_feature_group_slices(model)
    contributions = {"bias": float(np.asarray(intercept).ravel()[0])}

    for group_name, sl in slices.items():
        if sl.stop <= sl.start:
            contributions[group_name] = 0.0
            continue
        # Sparse dot dense => (n_samples,)
        group_score = X_matrix[:, sl].dot(w[sl])
        contributions[group_name] = group_score.A1.tolist() if hasattr(group_score, "A1") else np.asarray(group_score).ravel().tolist()

    return contributions


def predict(model, X, embedding_model=None):
    """Tahmin ve p_phishing_final döndür.

    If the model was trained with embeddings, you can provide an `embedding_model`
    (e.g. SentenceTransformer) to avoid re-loading it on every call.
    """
    # Hybrid models require embeddings during inference as well.
    # Older artifacts may miss `embedding_model_name`, so we infer the need from
    # the classifier's expected feature size.
    if embedding_model is None:
        embedding_model_name = model.get("embedding_model_name") or None
        vectorizer = model.get("vectorizer")
        if vectorizer is not None:
            base_feature_count = len(vectorizer.get_feature_names_out()) + len(get_security_feature_names())
            expected_feature_count = getattr(model.get("classifier"), "n_features_in_", base_feature_count)
            needs_embeddings = expected_feature_count > base_feature_count
        else:
            needs_embeddings = bool(embedding_model_name)

        if needs_embeddings:
            embedding_model = get_embedding_model(embedding_model_name or "all-MiniLM-L6-v2")
    X_matrix, _, _ = _build_feature_matrix(
        X,
        vectorizer=model["vectorizer"],
        fit=False,
        use_embeddings=embedding_model is not None,
        embedding_model=embedding_model,
    )
    probability_model = model.get("calibrated_classifier") or model["classifier"]
    probabilities = probability_model.predict_proba(X_matrix)
    phishing_index = list(probability_model.classes_).index("phishing")
    threshold = float(model.get("threshold", 0.5))
    predictions = _labels_from_probabilities(probabilities[:, phishing_index].tolist(), threshold)
    return {
        "predictions": predictions,
        "probabilities": probabilities[:, phishing_index].tolist(),
    }


def predict_with_group_contributions(model, X, embedding_model=None) -> dict:
    """
    Predict like `predict()`, plus optional group-level contribution breakdown.
    For hybrid models, this answers "how much did embeddings contribute?" numerically.
    """
    # Reuse predict() logic to ensure embeddings are loaded when needed,
    # but we need the built matrix too; so we mirror the matrix build here.
    if embedding_model is None:
        embedding_model_name = model.get("embedding_model_name") or None
        vectorizer = model.get("vectorizer")
        if vectorizer is not None:
            base_feature_count = len(vectorizer.get_feature_names_out()) + len(get_security_feature_names())
            expected_feature_count = getattr(model.get("classifier"), "n_features_in_", base_feature_count)
            needs_embeddings = expected_feature_count > base_feature_count
        else:
            needs_embeddings = bool(embedding_model_name)

        if needs_embeddings:
            embedding_model = get_embedding_model(embedding_model_name or "all-MiniLM-L6-v2")

    X_matrix, _, _ = _build_feature_matrix(
        X,
        vectorizer=model["vectorizer"],
        fit=False,
        use_embeddings=embedding_model is not None,
        embedding_model=embedding_model,
    )
    probability_model = model.get("calibrated_classifier") or model["classifier"]
    probabilities = probability_model.predict_proba(X_matrix)
    phishing_index = list(probability_model.classes_).index("phishing")
    threshold = float(model.get("threshold", 0.5))
    predictions = _labels_from_probabilities(probabilities[:, phishing_index].tolist(), threshold)

    group_contrib = _try_get_group_contributions(model, X_matrix)
    return {
        "predictions": predictions,
        "probabilities": probabilities[:, phishing_index].tolist(),
        "group_contributions": group_contrib,
    }
