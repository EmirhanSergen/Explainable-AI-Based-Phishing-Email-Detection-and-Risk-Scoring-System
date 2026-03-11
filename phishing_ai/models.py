"""
Öncü model, ana model ve model karşılaştırması.
"""

from pathlib import Path


def train_pioneer_model(X_train, y_train):
    """
    Öncü model: TF-IDF + security features ile Logistic Regression.
    p_phishing_baseline üretir.
    """
    raise NotImplementedError("Öncü model eğitimi implement edilecek")


def train_main_model(X_train, y_train):
    """
    Ana model: TF-IDF + security features + p_phishing_baseline ile nihai classifier.
    """
    raise NotImplementedError("Ana model eğitimi implement edilecek")


def compare_models(X_train, y_train, X_test, y_test):
    """Naive Bayes, Logistic Regression, Random Forest karşılaştırması."""
    raise NotImplementedError("Model karşılaştırması implement edilecek")


def save_model(model, path: str):
    """Model kaydet."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # TODO: joblib/pickle ile kaydet


def load_model(path: str):
    """Model yükle."""
    raise NotImplementedError("Model yükleme implement edilecek")


def predict(model, X):
    """Tahmin ve p_phishing_final döndür."""
    raise NotImplementedError("Predict implement edilecek")
