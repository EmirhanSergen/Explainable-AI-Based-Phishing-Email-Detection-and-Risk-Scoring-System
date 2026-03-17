"""
FastAPI dependencies: model yükleme vb.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from phishing_ai.config import MAIN_MODEL_PATH
from phishing_ai.explain import get_top_indicators
from phishing_ai.features import extract_security_features
from phishing_ai.models import load_model, predict
from phishing_ai.risk import compute_risk_score, get_risk_components, get_risk_level


class EmailAnalyzer:
    """Inference wrapper used by the FastAPI layer."""

    def __init__(self, model):
        self.model = model

    def analyze(self, text: str) -> dict:
        prediction_result = predict(self.model, [text])
        probability = prediction_result["probabilities"][0]
        prediction = prediction_result["predictions"][0]
        security_features = extract_security_features(text)
        risk_components = get_risk_components(
            p_phishing_final=probability,
            url_count=security_features["url_count"],
            keyword_count=security_features["keyword_count"],
        )
        risk_score = round(
            compute_risk_score(
                p_phishing_final=probability,
                url_count=security_features["url_count"],
                keyword_count=security_features["keyword_count"],
            ),
            2,
        )
        indicators = get_top_indicators(self.model, text, top_n=5)
        return {
            "prediction": prediction,
            "probability": probability,
            "risk_score": risk_score,
            "risk_level": get_risk_level(risk_score),
            "risk_components": risk_components,
            "top_indicators_pos": indicators["positive"],
            "top_indicators_neg": indicators["negative"],
        }


@lru_cache
def get_analyzer():
    """Phishing analyzer (model + pipeline) döndür."""
    model_path = Path(MAIN_MODEL_PATH)
    if not model_path.exists():
        raise RuntimeError(
            f"Trained model not found at {model_path}. Run `python scripts/train.py` first."
        )
    return EmailAnalyzer(load_model(model_path))
