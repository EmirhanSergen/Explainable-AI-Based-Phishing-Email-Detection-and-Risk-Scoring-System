"""
Metin normalizasyonu, TF-IDF ve security feature çıkarımı.
"""

import re

import numpy as np
from phishing_ai.config import CRITICAL_KEYWORDS, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE

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
    """Kritik kelime sayısı."""
    text_lower = normalize_text(text)
    return sum(1 for kw in CRITICAL_KEYWORDS if kw in text_lower)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    text_lower = normalize_text(text)
    return any(keyword in text_lower for keyword in keywords)


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


def get_tfidf_vectorizer():
    """TF-IDF vektörizer tanımı (unigram/bigram)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        strip_accents="unicode",
        lowercase=True,
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
