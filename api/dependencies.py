"""
FastAPI dependencies: model yükleme vb.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

from fastapi import Query

from phishing_ai.config import (
    HYBRID_MODEL_PATH,
    HYBRID_MODEL_V2_PATH,
    HYBRID_MODEL_V3_PATH,
    MAIN_MODEL_PATH,
    MAIN_MODEL_V2_PATH,
    MAIN_MODEL_V3_PATH,
    MODEL_METRICS_PATH,
    TEST_SAMPLES_PATH,
)
from phishing_ai.explain import get_top_indicators
from phishing_ai.features import extract_security_features
from phishing_ai.models import load_model, predict_with_group_contributions
from phishing_ai.risk import W_KW, W_PROB, W_URL
from phishing_ai.risk import compute_risk_score, get_risk_components, get_risk_level


class EmailAnalyzer:
    """Inference wrapper used by the FastAPI layer."""

    def __init__(self, model):
        self.model = model
        self.embedding_model = None
        if self.model.get("embedding_model_name"):
            # Cache embedding model once for hybrid inference.
            from phishing_ai.features import get_embedding_model

            self.embedding_model = get_embedding_model(self.model["embedding_model_name"])

    def analyze(self, text: str) -> dict:
        risk_weights = {
            "w_prob": self.model.get("risk_weights", {}).get("w_prob", W_PROB),
            "w_url": self.model.get("risk_weights", {}).get("w_url", W_URL),
            "w_kw": self.model.get("risk_weights", {}).get("w_kw", W_KW),
        }
        risk_thresholds = self.model.get("risk_thresholds") or {}
        prediction_result = predict_with_group_contributions(
            self.model, [text], embedding_model=self.embedding_model
        )
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
                w_prob=risk_weights["w_prob"],
                w_url=risk_weights["w_url"],
                w_kw=risk_weights["w_kw"],
            ),
            2,
        )
        indicators = get_top_indicators(self.model, text, top_n=5)
        return {
            "prediction": prediction,
            "probability": probability,
            "risk_score": risk_score,
            "risk_level": get_risk_level(risk_score, thresholds=risk_thresholds),
            "risk_components": risk_components,
            "risk_weights": risk_weights,
            "risk_thresholds": risk_thresholds,
            "top_indicators_pos": indicators["positive"],
            "top_indicators_neg": indicators["negative"],
            "group_contributions": prediction_result.get("group_contributions"),
        }


@lru_cache
def _load_analyzer(model: str) -> EmailAnalyzer:
    if model == "hybrid":
        model_path = Path(HYBRID_MODEL_PATH)
        if not model_path.exists():
            raise RuntimeError(
                f"Hybrid model not found at {model_path}. Run `python scripts/train.py --phase2` first."
            )
        return EmailAnalyzer(load_model(model_path))
    if model == "main_v2":
        model_path = Path(MAIN_MODEL_V2_PATH)
        if not model_path.exists():
            raise RuntimeError(
                f"Main v2 model not found at {model_path}. Run `python scripts/train.py --v2` first."
            )
        return EmailAnalyzer(load_model(model_path))
    if model == "hybrid_v2":
        model_path = Path(HYBRID_MODEL_V2_PATH)
        if not model_path.exists():
            raise RuntimeError(
                f"Hybrid v2 model not found at {model_path}. Run `python scripts/train.py --v2 --phase2` first."
            )
        return EmailAnalyzer(load_model(model_path))
    if model == "main_v3":
        model_path = Path(MAIN_MODEL_V3_PATH)
        if not model_path.exists():
            raise RuntimeError(
                f"Main v3 model not found at {model_path}. Run `python scripts/train.py --v3` first."
            )
        return EmailAnalyzer(load_model(model_path))
    if model == "hybrid_v3":
        model_path = Path(HYBRID_MODEL_V3_PATH)
        if not model_path.exists():
            raise RuntimeError(
                f"Hybrid v3 model not found at {model_path}. Run `python scripts/train.py --v3 --phase2` first."
            )
        return EmailAnalyzer(load_model(model_path))

    model_path = Path(MAIN_MODEL_PATH)
    if not model_path.exists():
        raise RuntimeError(
            f"Trained model not found at {model_path}. Run `python scripts/train.py` first."
        )
    return EmailAnalyzer(load_model(model_path))


def get_analyzer(
    model: str = Query(
        "main",
        pattern="^(main|hybrid|main_v2|hybrid_v2|main_v3|hybrid_v3)$",
    ),
) -> EmailAnalyzer:
    """FastAPI dependency returning the selected analyzer."""
    return _load_analyzer(model)


def _read_json_file(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        raise RuntimeError(f"Required artifact not found at {file_path}. Run the training pipeline first.")
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_model_metrics() -> dict:
    """Load persisted training metrics for the available models."""
    return _read_json_file(MODEL_METRICS_PATH)


def load_test_samples() -> list[dict]:
    """Load persisted test split samples for demo usage in the web UI."""
    payload = _read_json_file(TEST_SAMPLES_PATH)
    return payload.get("samples", [])


def get_random_sample_email() -> dict:
    """Return a random saved test-set email for quick UI demos."""
    samples = load_test_samples()
    if not samples:
        raise RuntimeError("No saved test samples found. Run the training pipeline first.")
    sample = random.choice(samples)
    return {
        "text": sample.get("text", ""),
        "label": sample.get("label"),
    }
