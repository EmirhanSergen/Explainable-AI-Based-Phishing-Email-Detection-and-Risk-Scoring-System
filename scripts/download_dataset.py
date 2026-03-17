#!/usr/bin/env python3
"""Download the CEAS_08 phishing dataset from Kaggle."""

from __future__ import annotations

import sys
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile, BadZipFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KAGGLE_DATASET = "naserabdullahalam/phishing-email-dataset"
KAGGLE_FILE = "CEAS_08.csv"
KAGGLE_ZIP = "phishing-email-dataset.zip"


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
        "-p",
        str(destination_dir),
        "--force",
    ]


def download_dataset(
    data_dir: str | Path | None = None,
    kaggle_config_dir: str | Path | None = None,
) -> Path:
    """Download CEAS_08.csv into data/raw and return the local path."""
    directories = ensure_data_directories(data_dir)
    raw_dir = directories["raw_dir"]
    output_path = raw_dir / KAGGLE_FILE
    zip_path = raw_dir / KAGGLE_ZIP

    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    if output_path.exists() and output_path.stat().st_size == 0:
        # A previous failed attempt may leave an empty file behind.
        output_path.unlink(missing_ok=True)

    if not has_kaggle_credentials(kaggle_config_dir=kaggle_config_dir):
        raise RuntimeError(
            "Kaggle credentials are missing. Add ~/.kaggle/kaggle.json or set "
            "KAGGLE_USERNAME and KAGGLE_KEY."
        )

    # Download the full dataset ZIP, then extract the file reliably.
    zip_path.unlink(missing_ok=True)
    command = build_kaggle_download_command(destination_dir=raw_dir)
    subprocess.run(command, check=True)

    if not zip_path.exists():
        # Kaggle CLI may use a different zip name; try to find the newest zip.
        candidates = sorted(raw_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise RuntimeError("Kaggle download completed but no .zip file found in data/raw")
        zip_path = candidates[0]

    _extract_ceas_csv_from_zip(zip_path=zip_path, raw_dir=raw_dir)
    return output_path


def _extract_ceas_csv_from_zip(zip_path: Path, raw_dir: Path) -> None:
    """Extract CEAS_08.csv from a zip payload."""
    try:
        with ZipFile(zip_path) as zf:
            # Validate archive integrity (raises if corrupt).
            bad_member = zf.testzip()
            if bad_member is not None:
                raise RuntimeError(f"Corrupted zip member: {bad_member}")

            members = zf.namelist()
            target = None
            for name in members:
                if name.endswith(f"/{KAGGLE_FILE}") or name == KAGGLE_FILE:
                    target = name
                    break
            if not target:
                raise RuntimeError(f"ZIP does not contain {KAGGLE_FILE}. Members: {members[:10]}")
            zf.extract(target, path=raw_dir)

            extracted = raw_dir / target
            final_path = raw_dir / KAGGLE_FILE
            if extracted != final_path:
                extracted.replace(final_path)
    except (BadZipFile, EOFError, RuntimeError) as exc:
        raise RuntimeError(f"ZIP cannot be extracted: {zip_path}") from exc

if __name__ == "__main__":
    csv_path = download_dataset()
    print(f"Dataset ready at {csv_path}")
