"""
Dataset yükleme, temizleme ve train/test split.
"""


def load_dataset():
    """Dataset yükle (Hugging Face zefang-liu/phishing-email-dataset)."""
    raise NotImplementedError("Dataset yükleme implement edilecek")


def clean_text(text: str) -> str:
    """Metin temizleme."""
    if not text or not isinstance(text, str):
        return ""
    return text.strip()


def prepare_dataset(df):
    """Dataset hazırlığı: temizlik, split, p_phishing_baseline ekleme."""
    raise NotImplementedError("Dataset hazırlığı implement edilecek")


def get_train_test_split(df, test_size=0.2, random_state=42):
    """Train/test split."""
    raise NotImplementedError("Split implement edilecek")
