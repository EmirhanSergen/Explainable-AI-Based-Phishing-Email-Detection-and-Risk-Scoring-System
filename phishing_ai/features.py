"""
Metin normalizasyonu, TF-IDF ve security feature çıkarımı.
"""

import re
from phishing_ai.config import CRITICAL_KEYWORDS, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE


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
    text_lower = text.lower() if text else ""
    return sum(1 for kw in CRITICAL_KEYWORDS if kw in text_lower)


def extract_security_features(text: str) -> dict:
    """Security feature çıkarımı: url_count, has_url, keyword_count."""
    return {
        "url_count": extract_url_count(text),
        "has_url": has_url(text),
        "keyword_count": extract_keyword_count(text),
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
