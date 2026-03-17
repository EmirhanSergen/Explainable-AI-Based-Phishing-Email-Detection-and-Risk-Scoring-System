"""
API route tanımları.
"""

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import (
    EmailAnalyzer,
    get_analyzer,
    get_random_sample_email,
    load_model_metrics,
)
from api.schemas import AnalyzeRequest, AnalyzeResponse, SampleEmailResponse

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


@router.get("/sample_email", response_model=SampleEmailResponse)
def sample_email():
    """Return a random saved email from the persisted test split."""
    try:
        return get_random_sample_email()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/model_metrics")
def model_metrics():
    """Return persisted training metrics for the main and hybrid models."""
    try:
        return load_model_metrics()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
