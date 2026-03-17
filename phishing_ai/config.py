"""
Konfigürasyon ve sabitler.
"""

# Kritik kelime listesi (phishing göstergeleri)
# Not: Bu liste kural-motoru değil; keyword_count için geniş bir sinyal havuzu.
CRITICAL_KEYWORDS = [
    # Credential / auth
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
    "update password",
    "password reset",
    # Account / security status
    "account",
    "account verification",
    "verify your account",
    "security alert",
    "unusual activity",
    "suspicious activity",
    "locked",
    "locked out",
    "suspended",
    "deactivated",
    "disabled",
    "compromised",
    # Urgency / pressure
    "urgent",
    "immediately",
    "action required",
    "asap",
    "final notice",
    "final warning",
    "within 24 hours",
    "expires",
    "expire",
    # Call-to-action / delivery bait
    "click here",
    "click below",
    "click the link",
    "open attachment",
    "download",
    "view document",
    "review",
    "verify now",
    "confirm now",
    "update now",
    # Finance / billing
    "invoice",
    "payment",
    "billing",
    "refund",
    "transaction",
    "bank",
    "wire transfer",
    "gift card",
    # Common spoofed brands/roles (weak signal; used only for counting)
    "microsoft",
    "google",
    "paypal",
    "apple",
    "amazon",
    "it support",
    "helpdesk",
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
