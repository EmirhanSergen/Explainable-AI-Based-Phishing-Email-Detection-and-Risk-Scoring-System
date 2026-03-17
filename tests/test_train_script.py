"""Tests for the training pipeline script."""

import json
import unittest.mock as mock

import pandas as pd


def test_run_training_pipeline_trains_and_saves_model(tmp_path):
    from scripts.train import run_training_pipeline

    dataset_path = tmp_path / "CEAS_08.csv"
    model_path = tmp_path / "main_model.joblib"
    metrics_path = tmp_path / "metrics.json"
    samples_path = tmp_path / "test_samples.json"
    pd.DataFrame(
        {
            "subject": [
                "Verify your account",
                "Security alert",
                "Meeting agenda",
                "Project update",
            ],
            "body": [
                "Click here to reset password immediately",
                "Login now to confirm your credentials",
                "Agenda for tomorrow team meeting",
                "Please review the sprint summary",
            ],
            "label": [1, 1, 0, 0],
        }
    ).to_csv(dataset_path, index=False)

    report = run_training_pipeline(
        dataset_path=dataset_path,
        model_path=model_path,
        metrics_path=metrics_path,
        samples_path=samples_path,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert samples_path.exists()
    assert "model_metrics" in report
    assert "logistic_regression" in report["model_metrics"]
    assert json.loads(metrics_path.read_text())["main"]["logistic_regression"]["accuracy"] >= 0
    assert len(json.loads(samples_path.read_text())["samples"]) > 0


def test_run_phase2_hybrid_pipeline_saves_hybrid_model(tmp_path):
    from scripts.train import run_phase2_hybrid_pipeline

    class FakeEmbeddingModel:
        def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
            import numpy as np

            _ = (convert_to_numpy, show_progress_bar)
            return np.zeros((len(texts), 3), dtype=float)

    dataset_path = tmp_path / "CEAS_08.csv"
    hybrid_path = tmp_path / "main_model_hybrid.joblib"
    metrics_path = tmp_path / "metrics.json"
    pd.DataFrame(
        {
            "subject": ["Verify your account", "Meeting agenda", "Security alert", "Project update"],
            "body": ["Reset password now", "Team meeting notes", "Confirm login", "Sprint summary"],
            "label": [1, 0, 1, 0],
        }
    ).to_csv(dataset_path, index=False)

    report = run_phase2_hybrid_pipeline(
        dataset_path=dataset_path,
        model_path=hybrid_path,
        embedding_model=FakeEmbeddingModel(),
        metrics_path=metrics_path,
    )

    assert hybrid_path.exists()
    assert metrics_path.exists()
    assert "hybrid_metrics" in report
    assert "hybrid" in json.loads(metrics_path.read_text())


def test_run_v2_training_pipeline_trains_and_saves_versioned_model(tmp_path):
    from scripts.train import run_v2_training_pipeline

    dataset_path = tmp_path / "CEAS_08.csv"
    model_path = tmp_path / "main_model_v2.joblib"
    metrics_path = tmp_path / "metrics.json"
    samples_path = tmp_path / "test_samples.json"
    pd.DataFrame(
        {
            "subject": [
                "Verify your account",
                "Security alert",
                "Team lunch",
                "Sprint update",
                "Password expired",
                "Meeting agenda",
            ],
            "body": [
                "Click here to reset password immediately",
                "Login now to confirm your credentials",
                "Lunch tomorrow with the product team",
                "Please review the sprint summary",
                "Urgent verify your login now",
                "Agenda for tomorrow team meeting",
            ],
            "label": [1, 1, 0, 0, 1, 0],
        }
    ).to_csv(dataset_path, index=False)

    report = run_v2_training_pipeline(
        dataset_path=dataset_path,
        model_path=model_path,
        metrics_path=metrics_path,
        samples_path=samples_path,
    )

    assert model_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert "main_v2" in metrics
    assert "selected_threshold" in metrics["main_v2"]
    assert "brier_score" in metrics["main_v2"]
    assert report["saved_model_path"] == str(model_path)


def test_run_phase2_hybrid_v2_pipeline_saves_versioned_hybrid_model(tmp_path):
    from scripts.train import run_phase2_hybrid_v2_pipeline

    class FakeEmbeddingModel:
        def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
            import numpy as np

            _ = (convert_to_numpy, show_progress_bar)
            return np.asarray(
                [[1.0, 0.0] if "verify" in text or "password" in text else [0.0, 1.0] for text in texts],
                dtype=float,
            )

    dataset_path = tmp_path / "CEAS_08.csv"
    model_path = tmp_path / "main_model_hybrid_v2.joblib"
    metrics_path = tmp_path / "metrics.json"
    pd.DataFrame(
        {
            "subject": [
                "Verify your account",
                "Security alert",
                "Team lunch",
                "Sprint update",
                "Password expired",
                "Meeting agenda",
            ],
            "body": [
                "Click here to reset password immediately",
                "Login now to confirm your credentials",
                "Lunch tomorrow with the product team",
                "Please review the sprint summary",
                "Urgent verify your login now",
                "Agenda for tomorrow team meeting",
            ],
            "label": [1, 1, 0, 0, 1, 0],
        }
    ).to_csv(dataset_path, index=False)

    report = run_phase2_hybrid_v2_pipeline(
        dataset_path=dataset_path,
        model_path=model_path,
        embedding_model=FakeEmbeddingModel(),
        max_rows=6,
        metrics_path=metrics_path,
    )

    assert model_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert "hybrid_v2" in metrics
    assert "selected_threshold" in metrics["hybrid_v2"]
    assert "brier_score" in metrics["hybrid_v2"]
    assert report["saved_hybrid_model_path"] == str(model_path)


def test_run_v3_training_pipeline_produces_artifacts(tmp_path):
    """run_v3_training_pipeline produces main_v3 pkl and metrics."""
    from scripts.train import run_v3_training_pipeline

    dataset_path = tmp_path / "CEAS_08.csv"
    model_path = tmp_path / "main_model_v3.pkl"
    metrics_path = tmp_path / "metrics.json"
    samples_path = tmp_path / "test_samples.json"
    pd.DataFrame(
        {
            "subject": ["Verify account", "Meeting", "Password reset", "Sprint", "Login", "Agenda"],
            "body": [
                "Click to reset",
                "Team notes",
                "Urgent verify",
                "Sprint summary",
                "Confirm credentials",
                "Meeting notes",
            ],
            "label": [1, 0, 1, 0, 1, 0],
        }
    ).to_csv(dataset_path, index=False)

    def fake_load_hf(max_rows=None, **kw):
        return pd.DataFrame(
            {"text": ["HF phishing email", "HF safe email"], "label": ["phishing", "legitimate"]},
        )

    with mock.patch("phishing_ai.data.load_hf_phishing_email_dataset", fake_load_hf):
        report = run_v3_training_pipeline(
            dataset_path=dataset_path,
            model_path=model_path,
            hf_max_rows=100,
            metrics_path=metrics_path,
            samples_path=samples_path,
        )

    assert model_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert "main_v3" in metrics
    assert "selected_threshold" in metrics["main_v3"]
    assert report["saved_model_path"] == str(model_path)


def test_run_phase2_hybrid_v3_pipeline_produces_artifacts(tmp_path):
    """run_phase2_hybrid_v3_pipeline produces hybrid_v3 pkl and metrics."""
    from scripts.train import run_phase2_hybrid_v3_pipeline

    class FakeEmbeddingModel:
        def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
            import numpy as np

            return np.zeros((len(texts), 3), dtype=float)

    dataset_path = tmp_path / "CEAS_08.csv"
    model_path = tmp_path / "main_model_hybrid_v3.pkl"
    metrics_path = tmp_path / "metrics.json"
    pd.DataFrame(
        {
            "subject": ["Verify", "Meeting", "Reset", "Sprint", "Login", "Notes"],
            "body": ["Click", "Notes", "Urgent", "Summary", "Confirm", "Agenda"],
            "label": [1, 0, 1, 0, 1, 0],
        }
    ).to_csv(dataset_path, index=False)

    def fake_load_hf(max_rows=None, **kw):
        return pd.DataFrame(
            {"text": ["HF phish", "HF safe"], "label": ["phishing", "legitimate"]},
        )

    with mock.patch("phishing_ai.data.load_hf_phishing_email_dataset", fake_load_hf):
        report = run_phase2_hybrid_v3_pipeline(
            dataset_path=dataset_path,
            model_path=model_path,
            embedding_model=FakeEmbeddingModel(),
            hf_max_rows=100,
            metrics_path=metrics_path,
        )

    assert model_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert "hybrid_v3" in metrics
    assert "selected_threshold" in metrics["hybrid_v3"]
    assert report["saved_hybrid_model_path"] == str(model_path)
