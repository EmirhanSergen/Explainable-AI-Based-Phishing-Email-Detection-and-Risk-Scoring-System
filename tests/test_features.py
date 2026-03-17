"""Tests for phishing_ai.features module."""

import pytest
from phishing_ai.features import (
    get_tfidf_vectorizer,
    normalize_text,
    extract_url_count,
    has_url,
    extract_keyword_count,
    extract_security_features,
)


def test_normalize_text():
    assert normalize_text("  HELLO  World  ") == "hello world"
    assert normalize_text("") == ""


def test_extract_url_count():
    assert extract_url_count("no url") == 0
    assert extract_url_count("visit https://evil.com") == 1
    assert extract_url_count("a https://a.com b http://b.com") == 2


def test_has_url():
    assert has_url("no url") is False
    assert has_url("click https://link.com") is True


def test_extract_keyword_count():
    assert extract_keyword_count("random text") == 0
    assert extract_keyword_count("verify your account") >= 2


def test_extract_keyword_count_uses_keyword_boundaries():
    assert extract_keyword_count("our accountant sent the report") == 0


def test_extract_keyword_count_weights_strong_keywords_more_than_weak():
    weak_only = extract_keyword_count("please review your account summary")
    strong_only = extract_keyword_count("verify your password immediately")
    assert strong_only > weak_only


def test_extract_security_features():
    feats = extract_security_features(
        "Urgent: verify your account and reset password at https://x.com"
    )
    assert feats["url_count"] == 1
    assert feats["has_url"] is True
    assert feats["keyword_count"] >= 3
    assert feats["has_urgent_word"] is True
    assert feats["has_credential_word"] is True
    assert feats["has_account_word"] is True


def test_extract_security_features_uses_keyword_boundaries_for_boolean_flags():
    feats = extract_security_features("our accountant sent the monthly report")
    assert feats["has_account_word"] is False


def test_get_tfidf_vectorizer_uses_denoising_defaults():
    vectorizer = get_tfidf_vectorizer()
    assert vectorizer.stop_words == "english"
    assert vectorizer.sublinear_tf is True
    assert vectorizer.min_df >= 1
    assert 0 < vectorizer.max_df <= 1.0
