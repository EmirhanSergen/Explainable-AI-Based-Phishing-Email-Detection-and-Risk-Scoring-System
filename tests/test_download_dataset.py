"""Tests for Kaggle dataset download helpers."""

from pathlib import Path


def test_ensure_data_directories_creates_expected_paths(tmp_path):
    from scripts.download_dataset import ensure_data_directories

    directories = ensure_data_directories(tmp_path)

    assert directories["data_dir"] == tmp_path
    assert directories["raw_dir"] == tmp_path / "raw"
    assert directories["processed_dir"] == tmp_path / "processed"
    assert directories["raw_dir"].exists()
    assert directories["processed_dir"].exists()


def test_has_kaggle_credentials_accepts_environment_variables(monkeypatch, tmp_path):
    from scripts.download_dataset import has_kaggle_credentials

    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "secret")

    assert has_kaggle_credentials(kaggle_config_dir=tmp_path) is True


def test_has_kaggle_credentials_accepts_kaggle_json(tmp_path, monkeypatch):
    from scripts.download_dataset import has_kaggle_credentials

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "kaggle.json").write_text(
        '{"username": "user", "key": "secret"}',
        encoding="utf-8",
    )

    assert has_kaggle_credentials(kaggle_config_dir=kaggle_dir) is True


def test_build_kaggle_download_command_targets_ceas08_file(tmp_path):
    from scripts.download_dataset import build_kaggle_download_command

    command = build_kaggle_download_command(destination_dir=tmp_path)

    assert command[:3] == ["kaggle", "datasets", "download"]
    assert "naserabdullahalam/phishing-email-dataset" in command
    assert "CEAS_08.csv" in command
    assert "--unzip" in command
    assert str(tmp_path) in command


def test_download_dataset_raises_without_credentials(tmp_path, monkeypatch):
    from scripts.download_dataset import download_dataset

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    try:
        download_dataset(data_dir=tmp_path, kaggle_config_dir=tmp_path / ".kaggle")
    except RuntimeError as exc:
        assert "Kaggle credentials" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when Kaggle credentials are missing")
