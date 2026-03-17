"""Tests for API endpoints."""

from fastapi.testclient import TestClient

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
                "top_indicators_pos": [
                    {"word": "verify", "contribution": 0.24},
                ],
                "top_indicators_neg": [
                    {"word": "meeting", "contribution": -0.08},
                ],
            }

    app.dependency_overrides[get_analyzer] = lambda: StubAnalyzer()
    try:
        response = client.post("/analyze_email", json={"text": "verify your account"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == "phishing"
    assert payload["risk_components"]["s_prob"] == 91.0
    assert payload["top_indicators_pos"][0]["word"] == "verify"
    assert payload["top_indicators_neg"][0]["word"] == "meeting"
