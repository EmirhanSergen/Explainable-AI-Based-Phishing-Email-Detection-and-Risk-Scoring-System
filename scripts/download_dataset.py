#!/usr/bin/env python3
"""Download the CEAS_08 phishing dataset from Kaggle."""

from __future__ import annotations

import sys
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KAGGLE_DATASET = "naserabdullahalam/phishing-email-dataset"
KAGGLE_FILE = "CEAS_08.csv"


def ensure_data_directories(data_dir: str | Path | None = None) -> dict[str, Path]:
    """Create the expected local data directories."""
    base_dir = Path(data_dir) if data_dir else Path("data")
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {
        "data_dir": base_dir,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
    }


def has_kaggle_credentials(kaggle_config_dir: str | Path | None = None) -> bool:
    """Return True when Kaggle credentials are available via env vars or kaggle.json."""
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True

    config_dir = Path(kaggle_config_dir) if kaggle_config_dir else Path.home() / ".kaggle"
    return (config_dir / "kaggle.json").exists()


def build_kaggle_download_command(destination_dir: str | Path) -> list[str]:
    """Build the Kaggle CLI command used for downloading the CEAS_08 CSV."""
    return [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-f",
        KAGGLE_FILE,
        "-p",
        str(destination_dir),
        "--unzip",
    ]


def download_dataset(
    data_dir: str | Path | None = None,
    kaggle_config_dir: str | Path | None = None,
) -> Path:
    """Download CEAS_08.csv into data/raw and return the local path."""
    directories = ensure_data_directories(data_dir)
    raw_dir = directories["raw_dir"]
    output_path = raw_dir / KAGGLE_FILE

    if output_path.exists():
        return output_path

    if not has_kaggle_credentials(kaggle_config_dir=kaggle_config_dir):
        raise RuntimeError(
            "Kaggle credentials are missing. Add ~/.kaggle/kaggle.json or set "
            "KAGGLE_USERNAME and KAGGLE_KEY."
        )

    command = build_kaggle_download_command(destination_dir=raw_dir)
    subprocess.run(command, check=True)
    return output_path


if __name__ == "__main__":
    csv_path = download_dataset()
    print(f"Dataset ready at {csv_path}")
