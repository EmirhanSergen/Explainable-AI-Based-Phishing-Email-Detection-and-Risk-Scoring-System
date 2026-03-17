"""
API route tanımları.
"""

from fastapi import APIRouter, Depends

from api.dependencies import EmailAnalyzer, get_analyzer
from api.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter()


@router.get("/health")
def health_check():
    """Sağlık kontrolü."""
    return {"status": "ok"}


@router.post("/analyze_email", response_model=AnalyzeResponse)
def analyze_email(
    request: AnalyzeRequest,
    analyzer: EmailAnalyzer = Depends(get_analyzer),
):
    """
    E-posta analizi: prediction, probability, risk_score, risk_level,
    risk_components, top_indicators.
    """
    return analyzer.analyze(request.text)
