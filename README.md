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

**Live app:** https://leaf-doctor-27classes.streamlit.app/

---

## How the honesty works

| Feature | Source artifact |
|---|---|
| Trust badge (per-class AP) | `assets/per_class_ap.csv` |
| "Could also be…" look-alikes | `assets/confusion_pairs.csv` (colour cosine ≥ 0.98) |
| Headline metrics & overfitting note | `assets/metrics.json` |
| Reference leaf crops | one labelled crop per class in `assets/refs/` |

The weakest classes (e.g. *Tomato leaf late blight*, AP 0.18) and the look-alike confusions are not
hidden — they're surfaced as the model's own honest self-assessment.

---

## Run it locally

```bash
git clone https://github.com/Wobbly1212/leaf-doctor.git
cd leaf-doctor
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The trained detector ships in the repo at `model/best.pt` (YOLOv8s, ~22 MB) so the app runs
out of the box with no extra download. Sample inputs are in `Streamlit_Test_Images/`.

## Project layout

```
streamlit_app.py     # the Streamlit app
prepare_assets.py    # regenerates everything under assets/ from the training artifacts
model/best.pt        # trained YOLOv8s detector (27 classes)
assets/              # per-class AP, confusion pairs, metrics, and reference crops
```
