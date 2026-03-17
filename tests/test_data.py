"""Tests for phishing_ai.data module."""

import pandas as pd


def _fake_hf_iter():
    """Fake HF streaming iterator (no network)."""
    rows = [
        {"Email Text": "Click here to verify your account now", "Email Type": "Phishing Email"},
        {"Email Text": "Team meeting tomorrow at 3pm", "Email Type": "Safe Email"},
        {"Email Text": "Reset password immediately", "Email Type": "Phishing Email"},
        {"Email Text": "", "Email Type": "Safe Email"},
        {"Email Text": "  ", "Email Type": "Phishing Email"},
    ]
    for r in rows:
        yield r


def test_load_hf_phishing_email_dataset_monkeypatch(monkeypatch):
    """HF loader returns normalized text/label without network."""
    from phishing_ai.data import load_hf_phishing_email_dataset

    def fake_load(name, split="train", streaming=False, **kw):
        class FakeIter:
            def __iter__(self):
                return _fake_hf_iter()

        return FakeIter()

    monkeypatch.setattr("datasets.load_dataset", fake_load)

    df = load_hf_phishing_email_dataset(max_rows=10)
    assert "text" in df.columns and "label" in df.columns
    assert len(df) == 3
    assert set(df["label"]) <= {"phishing", "legitimate"}
    texts = df["text"].str.lower()
    assert any("click" in t for t in texts)


def test_load_combined_dataset_dedup_and_labels(tmp_path):
    """Combined dataset deduplicates and maps labels correctly."""
    import unittest.mock as mock

    from phishing_ai.data import load_combined_dataset

    ceas_path = tmp_path / "ceas.csv"
    pd.DataFrame(
        {"subject": ["A"], "body": ["Duplicate text here"], "label": [1]},
    ).to_csv(ceas_path, index=False)

    def fake_load_hf(max_rows=None, split="train", dataset_name=""):
        return pd.DataFrame(
            {"text": ["duplicate text here", "Unique HF email"], "label": ["legitimate", "phishing"]},
        )

    import phishing_ai.data as data_module

    with mock.patch.object(data_module, "load_hf_phishing_email_dataset", fake_load_hf):
        combined = load_combined_dataset(ceas_path=ceas_path, hf_max_rows=100, add_source=True)
    assert "source" in combined.columns
    assert "text" in combined.columns and "label" in combined.columns
    assert set(combined["label"]) <= {"phishing", "legitimate"}
    assert combined["source"].nunique() >= 1


def test_clean_text():
    from phishing_ai.data import clean_text

    assert clean_text("  hello  ") == "hello"
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_prepare_dataset_combines_subject_body_and_normalizes_labels():
    from phishing_ai.data import prepare_dataset

    df = pd.DataFrame(
        {
            "subject": ["Verify account", "Project update"],
            "body": [
                "Urgent action required. Reset password now.",
                "Please review the sprint notes.",
            ],
            "label": [1, 0],
        }
    )

    prepared = prepare_dataset(df)

    assert list(prepared.columns) == ["text", "label"]
    assert prepared.loc[0, "text"] == "Verify account Urgent action required. Reset password now."
    assert prepared.loc[1, "text"] == "Project update Please review the sprint notes."
    assert list(prepared["label"]) == ["phishing", "legitimate"]


def test_get_train_test_split_returns_text_and_label_partitions():
    from phishing_ai.data import get_train_test_split

    df = pd.DataFrame(
        {
            "text": [
                "verify your account urgently",
                "team lunch tomorrow",
                "reset password now",
                "attached meeting notes",
            ],
            "label": ["phishing", "legitimate", "phishing", "legitimate"],
        }
    )

    X_train, X_test, y_train, y_test = get_train_test_split(
        df,
        test_size=0.5,
        random_state=7,
    )

    assert len(X_train) == 2
    assert len(X_test) == 2
    assert sorted(set(y_train) | set(y_test)) == ["legitimate", "phishing"]
