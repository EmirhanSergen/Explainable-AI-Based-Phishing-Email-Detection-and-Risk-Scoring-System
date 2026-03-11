"""
API route tanımları.
"""

from fastapi import APIRouter

from api.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter()


@router.get("/health")
def health_check():
    """Sağlık kontrolü."""
    return {"status": "ok"}


@router.post("/analyze_email", response_model=AnalyzeResponse)
def analyze_email(request: AnalyzeRequest):
    """
    E-posta analizi: prediction, probability, risk_score, risk_level,
    risk_components, top_indicators.
    """
    # TODO: get_analyzer() ile analiz yap
    raise NotImplementedError("Analyze endpoint implement edilecek")
