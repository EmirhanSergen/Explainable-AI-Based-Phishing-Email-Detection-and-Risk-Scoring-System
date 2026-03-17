"""Tests for API endpoints."""

from fastapi.testclient import TestClient

from api import routes
from api.dependencies import get_analyzer
from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_analyze_email_returns_prediction_payload():
    class StubAnalyzer:
        def analyze(self, text: str):
            assert text
            return {
                "prediction": "phishing",
                "probability": 0.91,
                "risk_score": 84.0,
                "risk_level": "Critical",
                "risk_components": {
                    "s_prob": 91.0,
                    "s_url": 50.0,
                    "s_kw": 80.0,
                },
                "risk_weights": {
                    "w_prob": 0.4,
                    "w_url": 0.3,
                    "w_kw": 0.3,
                },
                "top_indicators_pos": [
                    {"word": "verify", "contribution": 0.24},
                ],
                "top_indicators_neg": [
                    {"word": "meeting", "contribution": -0.08},
                ],
                "group_contributions": {
                    "bias": 0.1,
                    "tfidf": [0.2],
                    "security": [0.0],
                    "embedding": [0.3],
                },
            }

    # get_analyzer now accepts a `model` query param; override must accept it too.
    app.dependency_overrides[get_analyzer] = lambda model="main": StubAnalyzer()
    try:
        response = client.post("/analyze_email", json={"text": "verify your account"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == "phishing"
    assert payload["risk_components"]["s_prob"] == 91.0
    assert payload["risk_weights"]["w_prob"] == 0.4
    assert payload["top_indicators_pos"][0]["word"] == "verify"
    assert payload["top_indicators_neg"][0]["word"] == "meeting"
    assert payload["group_contributions"]["embedding"][0] == 0.3


def test_analyze_email_supports_model_query_param():
    class StubAnalyzer:
        def __init__(self, name: str):
            self.name = name

        def analyze(self, text: str):
            assert text
            return {
                "prediction": "phishing" if self.name == "hybrid" else "legitimate",
                "probability": 0.5,
                "risk_score": 50.0,
                "risk_level": "Medium",
                "risk_components": {"s_prob": 50.0, "s_url": 0.0, "s_kw": 0.0},
                "risk_weights": {"w_prob": 0.6, "w_url": 0.25, "w_kw": 0.15},
                "top_indicators_pos": [],
                "top_indicators_neg": [],
            }

    app.dependency_overrides[get_analyzer] = lambda model="main": StubAnalyzer(model)
    try:
        r1 = client.post("/analyze_email?model=main", json={"text": "x"})
        r2 = client.post("/analyze_email?model=hybrid", json={"text": "x"})
    finally:
        app.dependency_overrides.clear()

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["prediction"] == "legitimate"
    assert r2.json()["prediction"] == "phishing"


def test_analyze_email_supports_v2_model_query_param():
    class StubAnalyzer:
        def __init__(self, name: str):
            self.name = name

        def analyze(self, text: str):
            assert text
            return {
                "prediction": "phishing" if self.name.endswith("_v2") else "legitimate",
                "probability": 0.73,
                "risk_score": 62.0,
                "risk_level": "High",
                "risk_components": {"s_prob": 73.0, "s_url": 25.0, "s_kw": 40.0},
                "risk_weights": {"w_prob": 0.4, "w_url": 0.3, "w_kw": 0.3},
                "top_indicators_pos": [],
                "top_indicators_neg": [],
            }

    app.dependency_overrides[get_analyzer] = lambda model="main": StubAnalyzer(model)
    try:
        r1 = client.post("/analyze_email?model=main_v2", json={"text": "x"})
        r2 = client.post("/analyze_email?model=hybrid_v2", json={"text": "x"})
    finally:
        app.dependency_overrides.clear()

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["prediction"] == "phishing"
    assert r2.json()["prediction"] == "phishing"


def test_analyze_email_supports_v3_model_query_param():
    """main_v3 and hybrid_v3 are valid model query params (stub bypasses real load)."""
    class StubAnalyzer:
        def __init__(self, name: str):
            self.name = name

        def analyze(self, text: str):
            assert text
            return {
                "prediction": "phishing" if "v3" in self.name else "legitimate",
                "probability": 0.68,
                "risk_score": 58.0,
                "risk_level": "High",
                "risk_components": {"s_prob": 68.0, "s_url": 20.0, "s_kw": 35.0},
                "risk_weights": {"w_prob": 0.6, "w_url": 0.25, "w_kw": 0.15},
                "top_indicators_pos": [],
                "top_indicators_neg": [],
            }

    app.dependency_overrides[get_analyzer] = lambda model="main": StubAnalyzer(model)
    try:
        r1 = client.post("/analyze_email?model=main_v3", json={"text": "x"})
        r2 = client.post("/analyze_email?model=hybrid_v3", json={"text": "x"})
    finally:
        app.dependency_overrides.clear()

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["prediction"] == "phishing"
    assert r2.json()["prediction"] == "phishing"


def test_sample_email_returns_random_saved_sample(monkeypatch):
    monkeypatch.setattr(
        routes,
        "get_random_sample_email",
        lambda: {
            "text": "Reset your password now",
            "label": "phishing",
        },
    )

    response = client.get("/sample_email")

    assert response.status_code == 200
    assert response.json()["text"] == "Reset your password now"
    assert response.json()["label"] == "phishing"


def test_model_metrics_returns_saved_metrics(monkeypatch):
    monkeypatch.setattr(
        routes,
        "load_model_metrics",
        lambda: {
            "main": {"logistic_regression": {"accuracy": 0.9}},
            "hybrid": {"logistic_regression_hybrid": {"accuracy": 0.93}},
        },
    )

    response = client.get("/model_metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["main"]["logistic_regression"]["accuracy"] == 0.9
    assert payload["hybrid"]["logistic_regression_hybrid"]["accuracy"] == 0.93
