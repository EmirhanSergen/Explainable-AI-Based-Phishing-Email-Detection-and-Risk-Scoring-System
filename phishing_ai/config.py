"""
Konfigürasyon ve sabitler.
"""

# Kritik kelime listeleri (phishing göstergeleri)
# Strong keywords doğrudan credential/pressure/call-to-action sinyali taşır.
STRONG_CRITICAL_KEYWORDS = [
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
]

# Weak keywords bağlama göre meşru maillerde de geçebilir; daha düşük ağırlık alırlar.
WEAK_CRITICAL_KEYWORDS = [
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
CRITICAL_KEYWORDS = STRONG_CRITICAL_KEYWORDS + WEAK_CRITICAL_KEYWORDS

# Risk skoru ağırlıkları
W_PROB = 0.6
W_URL = 0.25
W_KW = 0.15
V2_RISK_WEIGHT_GRID = (
    {"w_prob": 0.6, "w_url": 0.25, "w_kw": 0.15},
    {"w_prob": 0.5, "w_url": 0.25, "w_kw": 0.25},
    {"w_prob": 0.4, "w_url": 0.3, "w_kw": 0.3},
)

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
HYBRID_MODEL_PATH = "models/main_model_hybrid.pkl"
MAIN_MODEL_V1_PATH = "models/main_model_v1.pkl"
HYBRID_MODEL_V1_PATH = "models/main_model_hybrid_v1.pkl"
MAIN_MODEL_V2_PATH = "models/main_model_v2.pkl"
HYBRID_MODEL_V2_PATH = "models/main_model_hybrid_v2.pkl"
MAIN_MODEL_V3_PATH = "models/main_model_v3.pkl"
HYBRID_MODEL_V3_PATH = "models/main_model_hybrid_v3.pkl"
MODEL_METRICS_PATH = "models/metrics.json"
TFIDF_VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
FEATURE_PIPELINE_PATH = "models/feature_pipeline.pkl"

# Persisted dataset artifacts
TEST_SAMPLES_PATH = "data/processed/test_samples.json"

# TF-IDF parametreleri
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_STOP_WORDS = "english"
TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 0.95
TFIDF_SUBLINEAR_TF = True
TFIDF_V2_PARAM_GRID = (
    {
        "stop_words": TFIDF_STOP_WORDS,
        "min_df": TFIDF_MIN_DF,
        "max_df": TFIDF_MAX_DF,
        "sublinear_tf": TFIDF_SUBLINEAR_TF,
    },
    {
        "stop_words": TFIDF_STOP_WORDS,
        "min_df": 2,
        "max_df": TFIDF_MAX_DF,
        "sublinear_tf": TFIDF_SUBLINEAR_TF,
    },
)

V2_MIN_PRECISION = 0.7
