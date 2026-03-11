"""
Konfigürasyon ve sabitler.
"""

# Kritik kelime listesi (phishing göstergeleri)
CRITICAL_KEYWORDS = [
    "verify",
    "password",
    "login",
    "urgent",
    "click here",
    "account",
    "security alert",
    "suspended",
    "confirm",
    "update",
    "verify your account",
    "reset password",
    "click below",
    "immediately",
    "action required",
]

# Risk skoru ağırlıkları
W_PROB = 0.6
W_URL = 0.25
W_KW = 0.15

# Risk seviye eşikleri (0-100)
RISK_THRESHOLDS = {
    "low": (0, 25),
    "medium": (26, 50),
    "high": (51, 75),
    "critical": (76, 100),
}

# Model dosya yolları
MODELS_DIR = "models"
PIONEER_MODEL_PATH = "models/pioneer_model.pkl"
MAIN_MODEL_PATH = "models/main_model.pkl"
TFIDF_VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
FEATURE_PIPELINE_PATH = "models/feature_pipeline.pkl"

# TF-IDF parametreleri
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
