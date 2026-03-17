"""
Öncü model, ana model ve model karşılaştırması.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from phishing_ai.features import (
    build_embedding_matrix,
    build_security_feature_matrix,
    get_security_feature_names,
    get_tfidf_vectorizer,
    normalize_text,
)


def _prepare_texts(texts) -> list[str]:
    return [normalize_text(text) for text in texts]


def _build_feature_matrix(
    texts,
    vectorizer=None,
    fit=False,
    use_embeddings=False,
    embedding_model=None,
):
    prepared_texts = _prepare_texts(texts)
    tfidf_vectorizer = vectorizer or get_tfidf_vectorizer()
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


def _build_artifact(classifier, vectorizer, embedding_model=None, embedding_dims: int = 0) -> dict:
    tfidf_feature_names = vectorizer.get_feature_names_out().tolist()
    embedding_feature_names = [
        f"embedding_{index}" for index in range(embedding_dims)
    ]
    return {
        "classifier": classifier,
        "vectorizer": vectorizer,
        "embedding_model": embedding_model,
        "security_feature_names": get_security_feature_names(),
        "feature_names": tfidf_feature_names
        + get_security_feature_names()
        + embedding_feature_names,
    }


def _metric_dict(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="phishing", zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label="phishing", zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label="phishing", zero_division=0),
    }


def train_pioneer_model(X_train, y_train):
    """
    Öncü model: TF-IDF + security features ile Logistic Regression.
    p_phishing_baseline üretir.
    """
    X_matrix, vectorizer, _ = _build_feature_matrix(X_train, fit=True)
    classifier = LogisticRegression(max_iter=1000, random_state=42)
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
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
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
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_matrix, y_train)
    embedding_dims = getattr(X_matrix, "shape", (0, 0))[1] - (
        len(vectorizer.get_feature_names_out()) + len(get_security_feature_names())
    )
    return _build_artifact(
        classifier,
        vectorizer,
        embedding_model=fitted_embedding_model,
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
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train_matrix, y_train)
    predictions = classifier.predict(X_test_matrix)
    return {"logistic_regression_hybrid": _metric_dict(y_test, predictions)}


def save_model(model, path: str):
    """Model kaydet."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    """Model yükle."""
    return joblib.load(path)


def predict(model, X):
    """Tahmin ve p_phishing_final döndür."""
    X_matrix, _, _ = _build_feature_matrix(
        X,
        vectorizer=model["vectorizer"],
        fit=False,
        use_embeddings=model.get("embedding_model") is not None,
        embedding_model=model.get("embedding_model"),
    )
    probabilities = model["classifier"].predict_proba(X_matrix)
    phishing_index = list(model["classifier"].classes_).index("phishing")
    predictions = model["classifier"].predict(X_matrix).tolist()
    return {
        "predictions": predictions,
        "probabilities": probabilities[:, phishing_index].tolist(),
    }
