from __future__ import annotations

import os
import subprocess
from pathlib import Path
import zipfile


OUT_DIR = Path("data/raw/kvasir_seg")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "debeshjha1/kvasirseg"


def have_kaggle_auth() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def main() -> None:
    if not have_kaggle_auth():
        raise RuntimeError(
            "Kaggle auth not found. Put kaggle.json in %USERPROFILE%\\.kaggle\\kaggle.json "
            "or set KAGGLE_USERNAME and KAGGLE_KEY env vars."
        )

    print("Downloading Kaggle dataset:", DATASET)
    cmd = ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(OUT_DIR), "--force"]
    subprocess.check_call(cmd)

    zip_files = list(OUT_DIR.glob("*.zip"))
    if not zip_files:
        raise RuntimeError("No zip downloaded. Check Kaggle CLI output.")

    zip_path = zip_files[0]
    print("Unzipping:", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(OUT_DIR)

    print("Done. Top level entries:")
    for p in OUT_DIR.iterdir():
        print("-", p.name)


if __name__ == "__main__":
    main()