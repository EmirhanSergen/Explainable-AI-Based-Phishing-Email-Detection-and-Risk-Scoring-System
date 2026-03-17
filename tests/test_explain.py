"""Tests for model explainability helpers."""

from phishing_ai.models import train_main_model

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


def test_get_top_indicators_returns_positive_and_negative_lists():
    from phishing_ai.explain import get_top_indicators

    model = train_main_model(TRAIN_TEXTS, TRAIN_LABELS)
    indicators = get_top_indicators(
        model,
        "urgent verify your account before the team meeting tomorrow",
        top_n=3,
    )

    assert set(indicators.keys()) == {"positive", "negative"}
    assert len(indicators["positive"]) <= 3
    assert len(indicators["negative"]) <= 3
    assert all(item["contribution"] > 0 for item in indicators["positive"])
    assert all(item["contribution"] < 0 for item in indicators["negative"])


def test_get_global_explanations_returns_ranked_features():
    from phishing_ai.explain import get_global_explanations

    model = train_main_model(TRAIN_TEXTS, TRAIN_LABELS)
    ranked = get_global_explanations(model, top_n=5)

    assert len(ranked["positive"]) <= 5
    assert len(ranked["negative"]) <= 5
    assert ranked["positive"][0]["feature"]
