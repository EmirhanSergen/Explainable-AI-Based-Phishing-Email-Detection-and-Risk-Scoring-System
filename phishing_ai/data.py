"""
Dataset yükleme, temizleme ve train/test split.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TEXT_COLUMN_CANDIDATES = ("text", "body", "message", "email", "content")
SUBJECT_COLUMN_CANDIDATES = ("subject", "title")
LABEL_COLUMN_CANDIDATES = ("label", "class", "target")


def _find_first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lowered_to_original = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lowered_to_original:
            return lowered_to_original[candidate]
    return None


def _normalize_label(value) -> str | None:
    if pd.isna(value):
        return None

    normalized = str(value).strip().lower()
    phishing_values = {"1", "true", "spam", "phishing", "phish", "malicious"}
    legitimate_values = {"0", "false", "ham", "legitimate", "safe", "benign"}

    if normalized in phishing_values:
        return "phishing"
    if normalized in legitimate_values:
        return "legitimate"
    return None


def load_dataset(csv_path: str | Path = "data/raw/CEAS_08.csv") -> pd.DataFrame:
    """Load the local CEAS_08 CSV and normalize it into text/label columns."""
    # CEAS_08.csv may vary by encoding and delimiter/quoting rules depending on export.
    # We try common encodings + delimiter sniffing, then fall back to a permissive parser.
    last_error: Exception | None = None
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")

    csv_path = Path(csv_path)
    sanitized_path = csv_path

    # Some CSV exports contain NUL bytes; the python csv engine cannot handle them.
    # Create a sanitized copy in-place next to the raw file.
    try:
        raw_bytes = csv_path.read_bytes()
        if b"\x00" in raw_bytes:
            sanitized_path = csv_path.with_suffix(".sanitized.csv")
            sanitized_path.write_bytes(raw_bytes.replace(b"\x00", b""))
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    for encoding in encodings:
        try:
            df = pd.read_csv(
                sanitized_path,
                encoding=encoding,
                sep=None,  # type: ignore[arg-type]
                engine="python",  # enables sep=None sniffing
                on_bad_lines="skip",
            )
            return prepare_dataset(df)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
            continue

    # Last resort: replace invalid characters and keep going with python engine.
    try:
        df = pd.read_csv(
            sanitized_path,
            encoding="utf-8",
            encoding_errors="replace",  # type: ignore[arg-type]
            sep=None,  # type: ignore[arg-type]
            engine="python",
            on_bad_lines="skip",
        )
        return prepare_dataset(df)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to parse dataset at {csv_path}. Last error: {last_error}. Final error: {exc}"
        )


def clean_text(text: str) -> str:
    """Metin temizleme."""
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the input dataset to a clean text/label dataframe."""
    label_column = _find_first_column(df, LABEL_COLUMN_CANDIDATES)
    if not label_column:
        raise ValueError("Dataset must contain a label column")

    text_column = _find_first_column(df, TEXT_COLUMN_CANDIDATES)
    subject_column = _find_first_column(df, SUBJECT_COLUMN_CANDIDATES)

    if not text_column and not subject_column:
        raise ValueError("Dataset must contain a body/text or subject column")

    if text_column:
        body_series = df[text_column].fillna("").map(clean_text)
    else:
        body_series = pd.Series([""] * len(df), index=df.index, dtype="object")

    if subject_column:
        subject_series = df[subject_column].fillna("").map(clean_text)
        text_series = (subject_series + " " + body_series).map(clean_text)
    else:
        text_series = body_series

    normalized_labels = df[label_column].map(_normalize_label)
    prepared = pd.DataFrame({"text": text_series, "label": normalized_labels})
    prepared = prepared[prepared["text"] != ""]
    prepared = prepared[prepared["label"].notna()]
    return prepared.reset_index(drop=True)


def get_train_test_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    """Train/test split."""
    stratify_labels = None
    label_counts = df["label"].value_counts()
    n_classes = df["label"].nunique()
    if isinstance(test_size, float):
        n_test = int(round(len(df) * test_size))
    else:
        n_test = int(test_size)

    if not label_counts.empty and label_counts.min() >= 2 and n_test >= n_classes:
        stratify_labels = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )
