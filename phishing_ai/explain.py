"""
SHAP entegrasyonu: global ve lokal analiz.
"""

from __future__ import annotations


def _get_linear_coefficients(model) -> tuple[list[str], list[float]]:
    classifier = model["classifier"]
    feature_names = model["feature_names"]
    if not hasattr(classifier, "coef_"):
        raise ValueError("Only linear models are currently supported for explanations")
    return feature_names, classifier.coef_[0].tolist()


def get_global_explanations(model, top_n: int = 10):
    """Return the most positive and negative globally weighted features."""
    feature_names, coefficients = _get_linear_coefficients(model)
    feature_weights = list(zip(feature_names, coefficients))
    positive = sorted(feature_weights, key=lambda item: item[1], reverse=True)[:top_n]
    negative = sorted(feature_weights, key=lambda item: item[1])[:top_n]
    return {
        "positive": [
            {"feature": feature, "weight": weight} for feature, weight in positive
        ],
        "negative": [
            {"feature": feature, "weight": weight} for feature, weight in negative
        ],
    }


def get_local_explanations(model, X_single, feature_names):
    """
    Lokal SHAP: tek e-posta için en önemli kelimeleri listeleyen fonksiyon.
    """
    _ = feature_names
    return get_top_indicators(model, X_single, top_n=10)


def get_top_indicators(model, text: str, top_n: int = 10) -> dict:
    """
    Return the most influential positive and negative token-level contributions.
    """
    vectorizer = model["vectorizer"]
    classifier = model["classifier"]
    tfidf_feature_names = vectorizer.get_feature_names_out().tolist()
    tfidf_vector = vectorizer.transform([text])
    tfidf_row = tfidf_vector.getrow(0)
    coefficients = classifier.coef_[0][: len(tfidf_feature_names)]

    contributions = []
    for feature_index, feature_value in zip(tfidf_row.indices, tfidf_row.data):
        contribution = float(feature_value * coefficients[feature_index])
        contributions.append((tfidf_feature_names[feature_index], contribution))

    positive = sorted(
        [item for item in contributions if item[1] > 0],
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]
    negative = sorted([item for item in contributions if item[1] < 0], key=lambda item: item[1])[
        :top_n
    ]

    return {
        "positive": [
            {"word": word, "contribution": contribution} for word, contribution in positive
        ],
        "negative": [
            {"word": word, "contribution": contribution} for word, contribution in negative
        ],
    }
