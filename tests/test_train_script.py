"""Tests for the training pipeline script."""

import pandas as pd


def test_run_training_pipeline_trains_and_saves_model(tmp_path):
    from scripts.train import run_training_pipeline

    dataset_path = tmp_path / "CEAS_08.csv"
    model_path = tmp_path / "main_model.joblib"
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

    report = run_training_pipeline(dataset_path=dataset_path, model_path=model_path)

    assert model_path.exists()
    assert "model_metrics" in report
    assert "logistic_regression" in report["model_metrics"]
