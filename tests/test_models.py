"""Tests for phishing_ai.models module."""

from pathlib import Path


TRAIN_TEXTS = [
    "urgent verify your account now at https://evil.com",
    "reset password immediately to avoid suspension",
    "security alert click here to confirm your login",
    "team meeting agenda for tomorrow morning",
    "please review the project status update",
    "lunch invitation for the product team",
]
TRAIN_LABELS = [
    "phishing",
    "phishing",
    "phishing",
    "legitimate",
    "legitimate",
    "legitimate",
]


def test_train_main_model_and_predict_returns_probabilities():
    from phishing_ai.models import predict, train_main_model

    model = train_main_model(TRAIN_TEXTS, TRAIN_LABELS)
    result = predict(
        model,
        [
            "urgent verify your password at https://bad.example",
            "agenda for the engineering sync tomorrow",
        ],
    )

    assert len(result["predictions"]) == 2
    assert len(result["probabilities"]) == 2
    assert all(0.0 <= prob <= 1.0 for prob in result["probabilities"])
    assert result["predictions"][0] == "phishing"


def test_compare_models_returns_metrics_for_expected_models():
    from phishing_ai.models import compare_models

    metrics = compare_models(
        TRAIN_TEXTS[:4],
        TRAIN_LABELS[:4],
        TRAIN_TEXTS[4:],
        TRAIN_LABELS[4:],
    )

    assert {"logistic_regression", "linear_svm", "naive_bayes"} <= set(metrics.keys())
    assert "f1" in metrics["logistic_regression"]


def test_save_and_load_model_round_trip(tmp_path):
    from phishing_ai.models import load_model, save_model, train_main_model

    model = train_main_model(TRAIN_TEXTS, TRAIN_LABELS)
    model_path = tmp_path / "main_model.joblib"

    save_model(model, model_path)
    loaded = load_model(model_path)

    assert loaded["classifier"].__class__ == model["classifier"].__class__
