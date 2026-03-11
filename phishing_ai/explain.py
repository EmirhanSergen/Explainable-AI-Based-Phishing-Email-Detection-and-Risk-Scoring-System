"""
SHAP entegrasyonu: global ve lokal analiz.
"""


def get_global_explanations(model, X, feature_names):
    """Global SHAP analizi: model genelinde hangi feature'lar önemli."""
    raise NotImplementedError("Global SHAP implement edilecek")


def get_local_explanations(model, X_single, feature_names):
    """
    Lokal SHAP: tek e-posta için en önemli kelimeleri listeleyen fonksiyon.
    """
    raise NotImplementedError("Lokal SHAP implement edilecek")


def get_top_indicators(model, text: str, top_n: int = 10) -> list:
    """
    Tek e-posta için en önemli kelimeleri contribution ile döndür.
    """
    raise NotImplementedError("Top indicators implement edilecek")
