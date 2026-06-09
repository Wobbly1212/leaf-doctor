"""
prepare_assets.py — one-time, local bundling step for the Leaf Doctor app.

The deployed app (Streamlit Cloud / Hugging Face Spaces) has NO access to the
local PlantDoc dataset, the `runs/` weights, or the `reports/` metrics. This
script makes the app self-contained by copying everything it needs into
`model/` and `assets/`:

  * model/best.pt            <- runs/run2/weights/best.pt   (the selected model)
  * assets/per_class_ap.csv  <- reports/per_class_test_ap.csv
  * assets/confusion_pairs.csv <- reports/similar_class_pairs.csv
  * assets/metrics.json      <- reports/final_metrics_summary.json
  * assets/classes.json      <- the 27 ordered names from data_proc/data.yaml
  * assets/refs/{id:02d}.jpg <- one real reference leaf crop per class

Run once from inside the app folder:  python prepare_assets.py
The outputs are committed to the repo and shipped to the cloud host.

Every number the app shows comes from these copied files — nothing is invented,
in keeping with the project's "no hallucinated numbers" rule (PlantDoc/CLAUDE.md).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import yaml

# --------------------------------------------------------------------------- #
# Paths. The live project lives one level up (note the trailing space in the
# parent folder name "ML_project "). The app folder is this file's directory.
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).resolve().parent
PROJECT = APP_DIR.parent / "PlantDoc"

SRC_MODEL = PROJECT / "runs" / "run2" / "weights" / "best.pt"
SRC_REPORTS = PROJECT / "reports"
SRC_DATA = PROJECT / "data_proc"
SRC_YAML = SRC_DATA / "data.yaml"

DST_MODEL = APP_DIR / "model" / "best.pt"
DST_ASSETS = APP_DIR / "assets"
DST_REFS = DST_ASSETS / "refs"

# crop selection: keep a clean, mid-sized example box per class
MIN_AREA, MAX_AREA = 0.05, 0.85
PAD = 0.08            # padding around the box, as a fraction of box size
LONG_SIDE = 256       # output thumbnail long side, px


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected source not found: {path}")
    return path


def copy_static() -> list[str]:
    """Copy the model and the three metric files; write classes.json."""
    DST_MODEL.parent.mkdir(parents=True, exist_ok=True)
    DST_ASSETS.mkdir(parents=True, exist_ok=True)

    shutil.copy2(_require(SRC_MODEL), DST_MODEL)

    copies = {
        "per_class_test_ap.csv": "per_class_ap.csv",
        "similar_class_pairs.csv": "confusion_pairs.csv",
        "final_metrics_summary.json": "metrics.json",
    }
    for src_name, dst_name in copies.items():
        shutil.copy2(_require(SRC_REPORTS / src_name), DST_ASSETS / dst_name)

    names = yaml.safe_load(_require(SRC_YAML).read_text())["names"]
    (DST_ASSETS / "classes.json").write_text(json.dumps(names, indent=2))

    out = [f"model/best.pt  ({DST_MODEL.stat().st_size / 1e6:.1f} MB)"]
    out += [f"assets/{n}" for n in copies.values()]
    out += [f"assets/classes.json  ({len(names)} classes)"]
    return out, names


def _iter_label_files(split: str):
    """Yield (label_path, image_path) pairs for a split, sorted for determinism."""
    lbl_dir = SRC_DATA / split / "labels"
    img_dir = SRC_DATA / split / "images"
    for lbl in sorted(lbl_dir.glob("*.txt")):
        img = img_dir / (lbl.stem + ".jpg")
        if img.exists():
            yield lbl, img


def _best_box_for(class_id: int):
    """Find the first clean, mid-sized box of `class_id`. Test split first, then train."""
    for split in ("test", "train"):
        for lbl, img in _iter_label_files(split):
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5 or int(float(parts[0])) != class_id:
                    continue
                _, xc, yc, w, h = (float(p) for p in parts)
                if MIN_AREA <= w * h <= MAX_AREA:
                    return img, (xc, yc, w, h)
    return None


def crop_reference(class_id: int) -> bool:
    """Crop and save one representative leaf for `class_id`. Returns True on success."""
    found = _best_box_for(class_id)
    if found is None:
        return False
    img_path, (xc, yc, w, h) = found

    im = cv2.imread(str(img_path))
    if im is None:
        return False
    H, W = im.shape[:2]

    # normalized box -> padded pixel box, clamped to the image
    bw, bh = w * W, h * H
    x1 = int(round((xc * W) - bw / 2 - PAD * bw))
    y1 = int(round((yc * H) - bh / 2 - PAD * bh))
    x2 = int(round((xc * W) + bw / 2 + PAD * bw))
    y2 = int(round((yc * H) + bh / 2 + PAD * bh))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return False

    crop = im[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    scale = LONG_SIDE / max(ch, cw)
    if scale < 1:
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)

    DST_REFS.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DST_REFS / f"{class_id:02d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True


def main() -> None:
    print("Bundling static files ...")
    done, names = copy_static()
    for line in done:
        print("  +", line)

    print(f"\nExtracting {len(names)} reference crops ...")
    missing = []
    for cid, name in enumerate(names):
        ok = crop_reference(cid)
        print(f"  {'+' if ok else '!'} {cid:02d}  {name}")
        if not ok:
            missing.append(name)

    n_ok = len(names) - len(missing)
    print(f"\nReference crops: {n_ok}/{len(names)} saved to assets/refs/")
    if missing:
        print("  WARNING: no suitable crop found for:", ", ".join(missing))
    print("\nDone. The app folder is now self-contained and ready to deploy.")


if __name__ == "__main__":
    main()
