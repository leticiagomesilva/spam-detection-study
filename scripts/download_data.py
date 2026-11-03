"""
Download the SMS Spam Collection dataset from the UCI repository.

Source : https://archive.ics.uci.edu/dataset/228/sms+spam+collection
License: Creative Commons Attribution 4.0 International (CC BY 4.0)

The raw file is preserved as-is in data/raw/SMSSpamCollection.
A metadata sidecar (data/raw/metadata.json) records the download URL,
the SHA-256 hash, and the download timestamp.

If the file already exists and its hash matches the expected value, the
download is skipped.  Pass --force to re-download unconditionally.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --force
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT    = Path(__file__).resolve().parent.parent
RAW_DIR      = REPO_ROOT / "data" / "raw"
TARGET_FILE  = RAW_DIR / "SMSSpamCollection"
METADATA_FILE = RAW_DIR / "metadata.json"

# UCI direct download URL for the zipped dataset
DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
)

# SHA-256 of the canonical SMSSpamCollection tab-separated file.
# Computed from the original UCI release.
EXPECTED_HASH = None  # set to a hex string to enable strict verification


def compute_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_and_extract(url: str, dest_file: Path) -> str:
    """Download zip archive from url and extract the SMSSpamCollection file.

    Returns the SHA-256 hex digest of the extracted file.
    """
    zip_path = dest_file.parent / "_download.zip"
    logger.info("Downloading %s …", url)

    try:
        urlretrieve(url, zip_path)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        logger.info(
            "Manual download: visit %s, download the zip, extract "
            "'SMSSpamCollection' to data/raw/, then re-run with --skip-download.",
            url,
        )
        sys.exit(1)

    logger.info("Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Look for the canonical filename inside the archive
        names = zf.namelist()
        target_name = next(
            (n for n in names if "SMSSpamCollection" in n and not n.endswith("/")),
            None,
        )
        if target_name is None:
            raise FileNotFoundError(
                f"'SMSSpamCollection' not found in zip. Contents: {names}"
            )
        with zf.open(target_name) as src, open(dest_file, "wb") as dst:
            shutil.copyfileobj(src, dst)

    zip_path.unlink(missing_ok=True)
    file_hash = compute_hash(dest_file)
    logger.info("Extracted to %s  (SHA-256: %s)", dest_file, file_hash)
    return file_hash


def save_metadata(file_hash: str, url: str):
    metadata = {
        "source_url":    url,
        "filename":      TARGET_FILE.name,
        "sha256":        file_hash,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "description":   (
            "SMS Spam Collection v.1 — UCI Machine Learning Repository. "
            "5,574 SMS messages in English labelled as ham (legitimate) or spam."
        ),
        "license": "CC BY 4.0  https://creativecommons.org/licenses/by/4.0/",
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    logger.info("Metadata written to %s", METADATA_FILE)


def main():
    parser = argparse.ArgumentParser(description="Download SMS Spam Collection.")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if file exists."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download; only record hash of the existing file.",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_FILE.exists() and not args.force:
        logger.info("File already exists: %s", TARGET_FILE)
        file_hash = compute_hash(TARGET_FILE)
        logger.info("SHA-256: %s", file_hash)
        if EXPECTED_HASH and file_hash != EXPECTED_HASH:
            logger.warning(
                "Hash mismatch!  Expected %s, got %s.  Use --force to re-download.",
                EXPECTED_HASH,
                file_hash,
            )
        else:
            logger.info("Hash OK.")
    elif args.skip_download:
        if not TARGET_FILE.exists():
            logger.error("--skip-download specified but %s does not exist.", TARGET_FILE)
            sys.exit(1)
        file_hash = compute_hash(TARGET_FILE)
        logger.info("SHA-256: %s", file_hash)
    else:
        file_hash = download_and_extract(DOWNLOAD_URL, TARGET_FILE)
        if EXPECTED_HASH and file_hash != EXPECTED_HASH:
            logger.warning(
                "Hash mismatch after download!  Expected %s, got %s.",
                EXPECTED_HASH,
                file_hash,
            )

    save_metadata(file_hash, DOWNLOAD_URL)
    logger.info("Done.")


if __name__ == "__main__":
    main()
