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
    top_indicators: list[TopIndicator]
