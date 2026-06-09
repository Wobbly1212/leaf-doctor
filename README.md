# 🌿 Leaf Doctor — Honest Second Opinion

An interactive plant-disease detector built on the **PlantDoc / YOLOv8** project. Upload a leaf
photo and the app does three things a normal classifier won't:

1. **Detects & names** each leaf's disease (YOLOv8s, 27 classes — the validation-selected *Run 2*).
2. **Tells you how much to trust it** — a TRUST badge calibrated from the model's *real* per-class
   test AP (a class the model scores 0.99 on reads green/Reliable; a 0.18 class reads red/Verify).
3. **Shows what else it might be** — a "could also be…" differential from the project's
   *pre-registered* colour-similarity confusion matrix, each with a reference leaf to compare.

Plus a downloadable diagnosis card and an optional "were we right?" check. Every number shown is
read from the project's own saved artifacts (`assets/`) — nothing is invented.

---

## Run locally

```bash
# 1. (once) bundle the model, metrics and reference crops from the parent project
python prepare_assets.py

# 2. install and launch
pip install -r requirements.txt
streamlit run streamlit_app.py
```

> `prepare_assets.py` is only needed if `model/best.pt` and `assets/` are not already present.
> It reads from the sibling `../PlantDoc` project. Once those files exist (and are committed),
> the app is fully self-contained and the script is not needed again.

## Deploy (shareable)

The folder is self-contained — `model/best.pt` (~22 MB) and `assets/` are committed, so no dataset
access is required at runtime.

**Streamlit Community Cloud**
1. Push this folder to a GitHub repo.
2. New app → pick the repo → main file = `streamlit_app.py`.
3. Advanced settings → Python **3.11**. Deploy.

**Hugging Face Spaces**
1. Create a Space → SDK = **Streamlit**.
2. Upload the folder contents (keep `streamlit_app.py` at the root).
3. `requirements.txt` is picked up automatically.

`requirements.txt` pins `opencv-python-headless` (cloud has no system GUI libs) and CPU-friendly
Torch; first model load takes a few seconds, then inference is ~1–2 s per image.

## Layout

```
streamlit_app.py     # the app
prepare_assets.py    # one-time bundling from ../PlantDoc
model/best.pt        # the selected YOLOv8s weights
assets/              # per_class_ap.csv, confusion_pairs.csv, metrics.json, classes.json, refs/*.jpg
.streamlit/config.toml
requirements.txt · runtime.txt
```

## How the honesty works

| Feature | Source artifact |
|---|---|
| Trust badge (per-class AP) | `reports/per_class_test_ap.csv` |
| "Could also be…" look-alikes | `reports/similar_class_pairs.csv` (colour cosine ≥ 0.98) |
| Headline metrics & overfitting note | `reports/final_metrics_summary.json` |
| Reference leaf crops | real labelled boxes from `data_proc/` |

The weakest classes (e.g. *Tomato leaf late blight*, AP 0.18) and the look-alike confusions are not
hidden — they're surfaced as the model's own honest self-assessment.
