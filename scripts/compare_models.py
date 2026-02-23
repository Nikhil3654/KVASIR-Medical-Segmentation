from __future__ import annotations

import json
from pathlib import Path
import csv


def load_best(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    best = d["best"]
    cfg = d["cfg"]
    return {"best": best, "cfg": cfg}


def main() -> None:
    run_dir = Path("outputs/runs")
    out_dir = Path("outputs/day4")
    out_dir.mkdir(parents=True, exist_ok=True)

    unet_path = run_dir / "unet_day3_summary.json"
    dl_path = run_dir / "deeplabv3_day4_summary.json"

    if not unet_path.exists():
        raise RuntimeError("Missing unet_day3_summary.json. Run UNet training or regenerate summary.")
    if not dl_path.exists():
        raise RuntimeError("Missing deeplabv3_day4_summary.json. Run python -m scripts.train_deeplabv3")

    unet = load_best(unet_path)
    dl = load_best(dl_path)

    rows = [
        {
            "model": "unet",
            "val_dice": unet["best"]["val_dice"],
            "val_iou": unet["best"]["val_iou"],
            "val_loss": unet["best"]["val_loss"],
            "image_size": unet["cfg"]["image_size"],
            "epochs": unet["cfg"]["epochs"],
            "batch_size": unet["cfg"]["batch_size"],
        },
        {
            "model": "deeplabv3_mobilenet",
            "val_dice": dl["best"]["val_dice"],
            "val_iou": dl["best"]["val_iou"],
            "val_loss": dl["best"]["val_loss"],
            "image_size": dl["cfg"]["image_size"],
            "epochs": dl["cfg"]["epochs"],
            "batch_size": dl["cfg"]["batch_size"],
        },
    ]

    out_json = out_dir / "day4_compare.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)

    out_csv = out_dir / "day4_compare.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("Saved:", out_json)
    print("Saved:", out_csv)
    for r in rows:
        print(r["model"], "dice", round(r["val_dice"], 4), "iou", round(r["val_iou"], 4), "loss", round(r["val_loss"], 4))


if __name__ == "__main__":
    main()