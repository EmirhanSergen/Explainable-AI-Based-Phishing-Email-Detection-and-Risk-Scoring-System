"""
Risk skoru hesaplama: risk_score_0_100 ve risk_level.
"""

from phishing_ai.config import W_PROB, W_URL, W_KW, RISK_THRESHOLDS


def compute_s_prob(p_phishing_final: float) -> float:
    """S_prob = 100 * p_phishing_final"""
    return 100 * p_phishing_final


def compute_s_url(url_count: int) -> float:
    """S_url = min(100, 25 * url_count)"""
    return min(100.0, 25 * url_count)


def compute_s_kw(keyword_count: int) -> float:
    """S_kw = min(100, 20 * keyword_count)"""
    return min(100.0, 20 * keyword_count)


def compute_risk_score(
    p_phishing_final: float,
    url_count: int,
    keyword_count: int,
    w_prob: float = W_PROB,
    w_url: float = W_URL,
    w_kw: float = W_KW,
) -> float:
    """
    risk_score_0_100 = w_prob * S_prob + w_url * S_url + w_kw * S_kw
    """
    s_prob = compute_s_prob(p_phishing_final)
    s_url = compute_s_url(url_count)
    s_kw = compute_s_kw(keyword_count)
    return w_prob * s_prob + w_url * s_url + w_kw * s_kw


def get_risk_components(p_phishing_final: float, url_count: int, keyword_count: int) -> dict:
    """Return the individual risk-score components for API/UI reporting."""
    return {
        "s_prob": compute_s_prob(p_phishing_final),
        "s_url": compute_s_url(url_count),
        "s_kw": compute_s_kw(keyword_count),
    }


def get_risk_level(risk_score: float) -> str:
    """
    risk_level: Low (0–25), Medium (26–50), High (51–75), Critical (76–100)
    """
    if risk_score <= 25:
        return "Low"
    if risk_score <= 50:
        return "Medium"
    if risk_score <= 75:
        return "High"
    return "Critical"
