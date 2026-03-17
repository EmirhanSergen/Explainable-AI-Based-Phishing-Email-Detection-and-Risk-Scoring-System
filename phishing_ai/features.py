"""
Metin normalizasyonu, TF-IDF ve security feature çıkarımı.
"""

from functools import lru_cache
import re

import numpy as np
from phishing_ai.config import (
    STRONG_CRITICAL_KEYWORDS,
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_STOP_WORDS,
    TFIDF_SUBLINEAR_TF,
    WEAK_CRITICAL_KEYWORDS,
)

URGENT_KEYWORDS = (
    "urgent",
    "immediately",
    "action required",
    "asap",
    "final notice",
    "final warning",
    "within 24 hours",
    "expires",
    "expire",
)
CREDENTIAL_KEYWORDS = (
    "password",
    "passcode",
    "otp",
    "2fa",
    "mfa",
    "login",
    "log in",
    "signin",
    "sign in",
    "username",
    "credentials",
    "verify",
    "confirm",
    "authenticate",
    "authentication",
    "reset password",
    "password reset",
)
ACCOUNT_KEYWORDS = (
    "account",
    "verify your account",
    "account verification",
    "security alert",
    "unusual activity",
    "suspicious activity",
    "locked",
    "locked out",
    "suspended",
    "deactivated",
    "disabled",
    "compromised",
    "billing",
    "invoice",
    "payment",
    "refund",
    "transaction",
    "bank",
    "wire transfer",
    "gift card",
)
SECURITY_FEATURE_NAMES = [
    "url_count",
    "has_url",
    "keyword_count",
    "has_urgent_word",
    "has_credential_word",
    "has_account_word",
]


def normalize_text(text: str) -> str:
    """Metin normalizasyonu: lowercase, temel temizlik."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_url_count(text: str) -> int:
    """URL sayısı."""
    url_pattern = r"https?://[^\s]+|www\.[^\s]+"
    return len(re.findall(url_pattern, text, re.IGNORECASE))


def has_url(text: str) -> bool:
    """URL var mı?"""
    return extract_url_count(text) > 0


def extract_keyword_count(text: str) -> int:
    """Weighted keyword count with regex boundaries to reduce false positives."""
    text_lower = normalize_text(text)
    score = 0
    for keyword in STRONG_CRITICAL_KEYWORDS:
        if _keyword_matches(text_lower, keyword):
            score += 2
    for keyword in WEAK_CRITICAL_KEYWORDS:
        if _keyword_matches(text_lower, keyword):
            score += 1
    return score


@lru_cache(maxsize=None)
def _compile_keyword_pattern(keyword: str) -> re.Pattern[str]:
    escaped_keyword = re.escape(keyword.lower()).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped_keyword}(?!\w)", re.IGNORECASE)


def _keyword_matches(text: str, keyword: str) -> bool:
    return bool(_compile_keyword_pattern(keyword).search(text))


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    text_lower = normalize_text(text)
    return any(_keyword_matches(text_lower, keyword) for keyword in keywords)


def extract_security_features(text: str) -> dict:
    """Security feature çıkarımı: url_count, has_url, keyword_count."""
    return {
        "url_count": extract_url_count(text),
        "has_url": has_url(text),
        "keyword_count": extract_keyword_count(text),
        "has_urgent_word": _contains_any(text, URGENT_KEYWORDS),
        "has_credential_word": _contains_any(text, CREDENTIAL_KEYWORDS),
        "has_account_word": _contains_any(text, ACCOUNT_KEYWORDS),
    }


def get_tfidf_vectorizer(**overrides):
    """TF-IDF vektörizer tanımı (unigram/bigram)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    params = {
        "max_features": TFIDF_MAX_FEATURES,
        "ngram_range": TFIDF_NGRAM_RANGE,
        "strip_accents": "unicode",
        "lowercase": True,
        "stop_words": TFIDF_STOP_WORDS,
        "min_df": TFIDF_MIN_DF,
        "max_df": TFIDF_MAX_DF,
        "sublinear_tf": TFIDF_SUBLINEAR_TF,
    }
    params.update(overrides)
    return TfidfVectorizer(
        **params,
    )


def get_security_feature_names() -> list[str]:
    """Return the ordered list of engineered feature names."""
    return SECURITY_FEATURE_NAMES.copy()


def build_security_feature_matrix(texts: list[str]) -> np.ndarray:
    """Convert security feature dicts into a dense numeric matrix."""
    matrix = []
    for text in texts:
        features = extract_security_features(text)
        matrix.append([float(features[name]) for name in SECURITY_FEATURE_NAMES])
    return np.asarray(matrix, dtype=float)


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazily load the optional sentence-transformers model."""
    from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]

    return SentenceTransformer(model_name)


def build_embedding_matrix(
    texts: list[str],
    embedding_model=None,
    model_name: str = "all-MiniLM-L6-v2",
):
    """Generate dense embedding vectors for the optional phase-2 experiment."""
    model = embedding_model or get_embedding_model(model_name=model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=float), model
