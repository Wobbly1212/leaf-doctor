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

## URL: https://leaf-doctor-27classes.streamlit.app/
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
