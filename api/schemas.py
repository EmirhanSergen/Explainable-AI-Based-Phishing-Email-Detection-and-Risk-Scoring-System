"""
Pydantic şemaları.
"""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """POST /analyze_email request."""

    text: str = Field(..., description="E-posta metni")


class RiskComponents(BaseModel):
    """Risk bileşenleri."""

    s_prob: float
    s_url: float
    s_kw: float


class RiskWeights(BaseModel):
    """Risk-score weight configuration used by the selected model."""

    w_prob: float
    w_url: float
    w_kw: float


class TopIndicator(BaseModel):
    """Öne çıkan kelime göstergesi."""

    word: str
    contribution: float


class AnalyzeResponse(BaseModel):
    """POST /analyze_email response."""

    prediction: str  # "phishing" | "legitimate"
    probability: float
    risk_score: float
    risk_level: str  # "Low" | "Medium" | "High" | "Critical"
    risk_components: RiskComponents
    risk_weights: RiskWeights
    top_indicators_pos: list[TopIndicator]
    top_indicators_neg: list[TopIndicator]
    group_contributions: dict | None = None


class SampleEmailResponse(BaseModel):
    """GET /sample_email response."""

    text: str
    label: str | None = None
