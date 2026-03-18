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
    w_prob: float | None = W_PROB,
    w_url: float | None = W_URL,
    w_kw: float | None = W_KW,
) -> float:
    """
    risk_score_0_100 = w_prob * S_prob + w_url * S_url + w_kw * S_kw
    """
    s_prob = compute_s_prob(p_phishing_final)
    s_url = compute_s_url(url_count)
    s_kw = compute_s_kw(keyword_count)
    prob_weight = W_PROB if w_prob is None else w_prob
    url_weight = W_URL if w_url is None else w_url
    kw_weight = W_KW if w_kw is None else w_kw
    return prob_weight * s_prob + url_weight * s_url + kw_weight * s_kw


def get_risk_components(p_phishing_final: float, url_count: int, keyword_count: int) -> dict:
    """Return the individual risk-score components for API/UI reporting."""
    return {
        "s_prob": compute_s_prob(p_phishing_final),
        "s_url": compute_s_url(url_count),
        "s_kw": compute_s_kw(keyword_count),
    }


def get_risk_level(risk_score: float, thresholds: dict | None = None) -> str:
    """
    risk_level: Low / Medium / High / Critical.

    Uses calibrated thresholds from the model artifact when provided,
    otherwise falls back to the hardcoded defaults (25 / 50 / 75).
    """
    t1 = (thresholds or {}).get("low_medium", 25)
    t2 = (thresholds or {}).get("medium_high", 50)
    t3 = (thresholds or {}).get("high_critical", 75)

    if risk_score < t1:
        return "Low"
    if risk_score < t2:
        return "Medium"
    if risk_score < t3:
        return "High"
    return "Critical"
