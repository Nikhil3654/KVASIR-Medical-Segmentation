from pathlib import Path
import csv


def test_pairs_csv_exists_and_has_header():
    p = Path("data/processed/pairs.csv")
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
    assert header == ["image_path", "mask_path"]