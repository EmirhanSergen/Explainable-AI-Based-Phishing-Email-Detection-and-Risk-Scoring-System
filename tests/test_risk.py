"""Tests for phishing_ai.risk module."""

import pytest
from phishing_ai.risk import (
    compute_s_prob,
    compute_s_url,
    compute_s_kw,
    compute_risk_score,
    get_risk_level,
)


def test_compute_s_prob():
    assert compute_s_prob(0.5) == 50.0
    assert compute_s_prob(0.87) == 87.0


def test_compute_s_url():
    assert compute_s_url(0) == 0
    assert compute_s_url(2) == 50
    assert compute_s_url(5) == 100


def test_compute_s_kw():
    assert compute_s_kw(0) == 0
    assert compute_s_kw(3) == 60
    assert compute_s_kw(6) == 100


def test_compute_risk_score():
    score = compute_risk_score(p_phishing_final=0.87, url_count=3, keyword_count=3)
    assert 0 <= score <= 100


def test_get_risk_level():
    assert get_risk_level(20) == "Low"
    assert get_risk_level(40) == "Medium"
    assert get_risk_level(60) == "High"
    assert get_risk_level(80) == "Critical"
