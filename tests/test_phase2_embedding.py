"""Tests for the optional phase-2 embedding experiment."""

import numpy as np

TRAIN_TEXTS = [
    "urgent verify your account now",
    "reset password immediately",
    "team lunch tomorrow",
    "project update attached",
]
TRAIN_LABELS = ["phishing", "phishing", "legitimate", "legitimate"]


class FakeEmbeddingModel:
    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        _ = (convert_to_numpy, show_progress_bar)
        rows = []
        for text in texts:
            rows.append(
                [
                    1.0 if "verify" in text or "password" in text else 0.0,
                    1.0 if "team" in text or "project" in text else 0.0,
                ]
            )
        return np.asarray(rows, dtype=float)


def test_train_embedding_hybrid_model_and_predict():
    from phishing_ai.models import predict, train_embedding_hybrid_model

    model = train_embedding_hybrid_model(
        TRAIN_TEXTS,
        TRAIN_LABELS,
        embedding_model=FakeEmbeddingModel(),
    )
    result = predict(model, ["verify password now", "team project update"])

    assert len(result["predictions"]) == 2
    assert all(0.0 <= prob <= 1.0 for prob in result["probabilities"])


def test_compare_models_with_embeddings_returns_hybrid_metrics():
    from phishing_ai.models import compare_models_with_embeddings

    metrics = compare_models_with_embeddings(
        TRAIN_TEXTS[:3],
        TRAIN_LABELS[:3],
        TRAIN_TEXTS[3:],
        TRAIN_LABELS[3:],
        embedding_model=FakeEmbeddingModel(),
    )

    assert "logistic_regression_hybrid" in metrics
    assert "accuracy" in metrics["logistic_regression_hybrid"]
