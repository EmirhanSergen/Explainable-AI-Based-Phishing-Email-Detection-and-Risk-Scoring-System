"""Tests for phishing_ai.data module."""

import pandas as pd


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
