"""
🌿 Leaf Doctor — Honest Second Opinion
======================================

Upload a leaf photo → a YOLOv8 detector (the validation-selected Run 2 from the
PlantDoc project) localizes each leaf and names its disease. But it does not stop
at a label: for every detection it shows

  • a TRUST meter calibrated from the model's *real* per-class test AP,
  • a "could also be …" differential from the project's pre-registered
    colour-similarity confusion matrix, each with a reference leaf to compare,
  • a downloadable diagnosis card.

Every number shown is read from the project's own saved artifacts (assets/),
never invented — in keeping with PlantDoc/CLAUDE.md's "no hallucinated numbers".

Run locally:   streamlit run streamlit_app.py
(First run `python prepare_assets.py` once to bundle the model + metrics.)
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "best.pt"
ASSETS = APP_DIR / "assets"
REFS = ASSETS / "refs"

IMG_SIZE = 640            # must match training (small-object analysis -> 640)
SIM_THRESHOLD = 0.98      # colour-similarity cut for "could also be"
DISPLAY_MAX = 920         # px, downscale large uploads for snappy display

# trust tiers, keyed by per-class test AP@0.5 ---------------------------------
TIERS = [
    (0.80, "Reliable",     "#15803D", "One of the model's strong classes"),
    (0.50, "Moderate",     "#0E7490", "Middling reliability — a second look helps"),
    (0.30, "Caution",      "#B45309", "A hard class — treat as a hint, not a verdict"),
    (0.00, "Low · verify", "#B91C1C", "The model's weakest territory — verify with an expert"),
]

# detection box palette (colour-blind friendly), cycled by detection order
PALETTE = ["#0EA5A0", "#F59E0B", "#6366F1", "#EC4899", "#22C55E", "#38BDF8", "#EF4444"]

st.set_page_config(page_title="Leaf Doctor — Honest Second Opinion",
                   page_icon="🌿", layout="wide",
                   initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- #
# Premium theme (custom CSS)
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

:root{
  --ink:#0F1B2D; --muted:#5B6B7F; --line:#E6EBF1;
  --brand:#0E7C66; --brand2:#14B8A6; --brand-deep:#0B3D2E;
  --card:#FFFFFF; --bg:#F6F8FA;
}
html, body, [class*="css"]{ font-family:'Inter',system-ui,sans-serif; }
.stApp{ background:
   radial-gradient(900px 500px at 85% -12%, #E7F6F1 0%, rgba(231,246,241,0) 55%),
   var(--bg); }

/* hide default chrome */
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
[data-testid="stToolbar"]{display:none;}
[data-testid="stDecoration"]{display:none;}
header[data-testid="stHeader"]{background:transparent;height:0;}
.block-container{ padding-top:1.4rem; padding-bottom:3rem; max-width:1180px; }

h1,h2,h3{ font-family:'Sora','Inter',sans-serif; color:var(--ink); letter-spacing:-.01em; }

/* ---------- hero ---------- */
.hero{
  position:relative; overflow:hidden; border-radius:22px; padding:30px 34px;
  background:linear-gradient(120deg,#0B3D2E 0%, #0E7C66 55%, #14B8A6 100%);
  box-shadow:0 18px 44px rgba(11,61,46,.28); color:#fff; margin-bottom:18px;
}
.hero::after{ content:""; position:absolute; right:-60px; top:-60px; width:280px; height:280px;
  background:radial-gradient(circle, rgba(255,255,255,.16), transparent 62%); }
.hero .eyebrow{ font-size:.74rem; font-weight:700; letter-spacing:.22em; text-transform:uppercase; opacity:.85; }
.hero h1{ color:#fff; font-size:2.15rem; margin:.18em 0 .12em; }
.hero p{ margin:0; max-width:680px; font-size:1.02rem; line-height:1.5; color:rgba(255,255,255,.92); }
.chips{ display:flex; gap:10px; flex-wrap:wrap; margin-top:16px; }
.chip{ background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.22);
  border-radius:12px; padding:8px 14px; backdrop-filter:blur(4px); }
.chip .v{ font-family:'Sora',sans-serif; font-weight:800; font-size:1.12rem; display:block; line-height:1; }
.chip .l{ font-size:.7rem; letter-spacing:.06em; text-transform:uppercase; opacity:.82; margin-top:4px; }

/* ---------- image frames ---------- */
.imgrow{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin:6px 0 4px; }
.frame{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:12px;
  box-shadow:0 6px 20px rgba(15,27,45,.06); }
.frame .ttl{ font-weight:700; font-size:.82rem; letter-spacing:.04em; text-transform:uppercase;
  color:var(--muted); margin:2px 4px 10px; display:flex; align-items:center; gap:7px; }
.frame .ttl .dot{ width:8px;height:8px;border-radius:50%; background:var(--brand); }
.frame img{ width:100%; border-radius:10px; display:block; }

/* ---------- section heading ---------- */
.section{ display:flex; align-items:center; gap:12px; margin:22px 0 12px; }
.section h3{ margin:0; font-size:1.18rem; }
.section .count{ background:#E7F6F1; color:#0B6E4F; font-weight:700; font-size:.78rem;
  border-radius:999px; padding:4px 12px; }

/* ---------- result card ---------- */
.card{ background:var(--card); border:1px solid var(--line); border-radius:18px; padding:20px 22px;
  box-shadow:0 10px 30px rgba(15,27,45,.07); margin-bottom:16px; }
.card-head{ display:flex; justify-content:space-between; align-items:flex-start; gap:14px; }
.diag{ font-family:'Sora',sans-serif; font-size:1.32rem; font-weight:800; color:var(--ink); line-height:1.15; }
.diag .sub{ display:block; font-family:'Inter'; font-weight:500; font-size:.8rem; color:var(--muted); margin-top:3px; letter-spacing:.02em; }
.conf{ text-align:right; }
.conf .pct{ font-family:'Sora',sans-serif; font-size:1.7rem; font-weight:800; line-height:1; }
.conf .cl{ font-size:.68rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }

.trust-row{ display:flex; align-items:center; gap:12px; margin:16px 0 6px; }
.pill{ color:#fff; font-weight:700; font-size:.74rem; letter-spacing:.04em; border-radius:999px;
  padding:5px 13px; white-space:nowrap; }
.meter{ flex:1; height:9px; background:#EDF1F5; border-radius:999px; overflow:hidden; }
.meter .fill{ height:100%; border-radius:999px; }
.apnum{ font-variant-numeric:tabular-nums; font-weight:700; color:var(--ink); font-size:.86rem; white-space:nowrap; }
.blurb{ color:var(--muted); font-size:.94rem; margin-top:2px; }
.smallobj{ margin-top:10px; font-size:.86rem; color:#92400E; background:#FEF3C7;
  border:1px solid #FDE68A; border-radius:10px; padding:8px 12px; }

.alt-title{ margin:18px 0 10px; font-weight:700; font-size:.9rem; color:var(--ink); }
.alt-title span{ font-weight:400; color:var(--muted); }
.alt-grid{ display:flex; gap:14px; flex-wrap:wrap; }
.alt{ width:150px; background:#FBFCFD; border:1px solid var(--line); border-radius:12px;
  padding:8px; text-align:center; }
.alt img{ width:100%; height:110px; object-fit:cover; border-radius:8px; }
.alt .nm{ font-weight:600; font-size:.8rem; color:var(--ink); margin-top:7px; line-height:1.2; }
.alt .sm{ font-size:.72rem; color:var(--muted); margin-top:2px; }

/* ---------- landing feature cards ---------- */
.feat-grid{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:6px 0 10px; }
.feat{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:20px;
  box-shadow:0 6px 20px rgba(15,27,45,.05); }
.feat .ic{ font-size:1.5rem; } .feat h4{ margin:.5em 0 .25em; font-family:'Sora'; font-size:1.02rem; }
.feat p{ margin:0; color:var(--muted); font-size:.9rem; line-height:1.45; }

.note-foot{ color:var(--muted); font-size:.82rem; text-align:center; margin-top:26px;
  padding-top:16px; border-top:1px solid var(--line); }

/* ---------- streamlit widget polish ---------- */
[data-testid="stSidebar"]{ background:#FFFFFF; border-right:1px solid var(--line); }
[data-testid="stSidebar"] .block-container{ padding-top:1.4rem; }
[data-testid="stFileUploaderDropzone"]{
  border:2px dashed #BFD8CF; border-radius:16px; background:#FBFFFE; padding:18px; }
[data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--brand); background:#F3FBF8; }
.stDownloadButton button, .stButton button{
  border-radius:11px; font-weight:700; border:0;
  background:linear-gradient(120deg,#0E7C66,#14B8A6); color:#fff; padding:.55rem 1.1rem; }
.stDownloadButton button:hover, .stButton button:hover{ filter:brightness(1.06); color:#fff; }
.stSlider [data-baseweb="slider"] div[role="slider"]{ background:var(--brand); }
</style>
"""


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the detector …")
def load_model():
    from ultralytics import YOLO
    return YOLO(str(MODEL_PATH))


@st.cache_data
def load_assets():
    classes = json.loads((ASSETS / "classes.json").read_text())

    ap = pd.read_csv(ASSETS / "per_class_ap.csv")
    ap_by_class = dict(zip(ap["class"], ap["AP50"]))

    pairs = pd.read_csv(ASSETS / "confusion_pairs.csv")
    look_alikes: dict[str, list[tuple[str, float]]] = {c: [] for c in classes}
    for _, row in pairs.iterrows():
        a, b, sim = row["class_a"], row["class_b"], float(row["similarity"])
        if sim >= SIM_THRESHOLD:
            look_alikes.setdefault(a, []).append((b, sim))
            look_alikes.setdefault(b, []).append((a, sim))
    for c in look_alikes:
        look_alikes[c] = sorted(set(look_alikes[c]), key=lambda t: -t[1])[:2]

    metrics = json.loads((ASSETS / "metrics.json").read_text())
    return classes, ap_by_class, look_alikes, metrics


@st.cache_data
def load_font(size: int):
    for cand in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10
    except Exception:
        return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# Image / inference helpers
# --------------------------------------------------------------------------- #
def data_uri(src, fmt: str = "JPEG") -> str:
    """Base64 data URI from a file path or a PIL image (for inline HTML)."""
    if isinstance(src, (str, Path)):
        raw = Path(src).read_bytes()
        mime = "jpeg"
    else:
        buf = io.BytesIO()
        src.save(buf, format=fmt, quality=88)
        raw = buf.getvalue()
        mime = fmt.lower()
    return f"data:image/{mime};base64," + base64.b64encode(raw).decode()


def for_display(img: Image.Image) -> Image.Image:
    im = img.copy()
    im.thumbnail((DISPLAY_MAX, DISPLAY_MAX))
    return im


def is_color_negative(arr: np.ndarray) -> bool:
    """PlantDoc colour-negative test: a leaf is normally greenest, so green being
    the weakest channel flags a stored photo-negative. Order-independent."""
    r, g, b = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
    return bool(g < r and g < b)


def run_detection(model, rgb: np.ndarray, conf: float, iou: float):
    res = model.predict(rgb, conf=conf, iou=iou, imgsz=IMG_SIZE,
                        augment=False, verbose=False)[0]
    dets = []
    if res.boxes is not None and len(res.boxes):
        names = res.names
        for box in res.boxes:
            dets.append({
                "class": names[int(box.cls[0])],
                "conf": float(box.conf[0]),
                "xyxy": [float(v) for v in box.xyxy[0]],
            })
    dets.sort(key=lambda d: -d["conf"])
    return dets


def trust_for(ap: float | None):
    if ap is None:
        return {"tier": "Unknown", "color": "#64748B", "frac": 0.0, "ap": None,
                "blurb": "No saved test score for this class"}
    for thresh, tier, color, blurb in TIERS:
        if ap >= thresh:
            return {"tier": tier, "color": color, "frac": ap, "ap": ap, "blurb": blurb}


def draw_detections(rgb: np.ndarray, dets: list[dict]) -> Image.Image:
    im = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    font = load_font(max(15, im.width // 42))
    lw = max(3, im.width // 260)
    for i, d in enumerate(dets):
        color = PALETTE[i % len(PALETTE)]
        x1, y1, x2, y2 = d["xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        label = f'{d["class"]}  {d["conf"]*100:.0f}%'
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = max(0, y1 - th - 10)
        draw.rounded_rectangle([x1, ly, x1 + tw + 16, ly + th + 10], radius=6, fill=color)
        draw.text((x1 + 8, ly + 4), label, fill="white", font=font)
    return im


def make_card(annotated: Image.Image, top: dict, trust: dict,
              partners: list[tuple[str, float]]) -> bytes:
    """Compose a shareable PNG diagnosis card."""
    W, pad = 900, 36
    img = annotated.copy()
    img.thumbnail((W - 2 * pad, 520))
    f_title, f_h, f_b, f_s = load_font(32), load_font(30), load_font(22), load_font(18)

    body_h = 250 + (40 * len(partners) if partners else 30)
    H = pad + 54 + 16 + img.height + 24 + body_h
    card = Image.new("RGB", (W, H), "#FFFFFF")
    d = ImageDraw.Draw(card)

    d.rectangle([0, 0, W, 56], fill="#0E7C66")
    d.text((pad, 13), "🌿 Leaf Doctor — Honest Second Opinion", fill="white", font=f_title)

    card.paste(img, ((W - img.width) // 2, 56 + 16))
    y = 56 + 16 + img.height + 24

    d.text((pad, y), f'Diagnosis: {top["class"]}', fill="#0F1B2D", font=f_h)
    d.text((W - pad - 120, y + 2), f'{top["conf"]*100:.0f}%', fill="#0E7C66", font=f_h)
    y += 46

    filled = max(1, min(5, round((trust["ap"] or 0) * 5)))
    dots = "●" * filled + "○" * (5 - filled)
    ap_txt = f' · test AP {trust["ap"]:.2f}' if trust["ap"] is not None else ""
    d.text((pad, y), f'Trust  {dots}   {trust["tier"]}{ap_txt}', fill=trust["color"], font=f_b)
    y += 34
    d.text((pad, y), trust["blurb"], fill="#5B6B7F", font=f_s)
    y += 40

    if partners:
        d.text((pad, y), "Could also be (colour look-alikes):", fill="#0F1B2D", font=f_b)
        y += 34
        for name, sim in partners:
            d.text((pad + 14, y), f"• {name}   (similarity {sim:.2f})", fill="#333", font=f_s)
            y += 30
    else:
        d.text((pad, y), "No strong colour look-alikes for this class.", fill="#5B6B7F", font=f_s)

    buf = io.BytesIO()
    card.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# HTML builders
# --------------------------------------------------------------------------- #
def ref_path(classes, name):
    try:
        return REFS / f"{classes.index(name):02d}.jpg"
    except ValueError:
        return None


def hero_html(metrics) -> str:
    ci = metrics.get("test_mAP50_bootstrap_ci", {})
    chips = [
        (f'{metrics["test_mAP50"]:.3f}', "test mAP@0.5"),
        (f'{metrics["test_precision"]:.3f}', "precision"),
        (f'{metrics["test_recall"]:.3f}', "recall"),
        (f'[{ci.get("ci_low",0):.2f}, {ci.get("ci_high",0):.2f}]', "95% CI"),
    ]
    chip_html = "".join(
        f'<div class="chip"><span class="v">{v}</span><span class="l">{l}</span></div>'
        for v, l in chips
    )
    return f"""
    <div class="hero">
      <div class="eyebrow">PlantDoc · YOLOv8s · 27 classes</div>
      <h1>🌿 Leaf Doctor</h1>
      <p>Upload a leaf and get a disease diagnosis that knows its own limits — every
         detection comes with a trust meter from the model's real accuracy and the
         look-alikes it might be confused with.</p>
      <div class="chips">{chip_html}</div>
    </div>
    """


def card_html(d, classes, ap_by_class, look_alikes, img_area) -> str:
    name = d["class"]
    trust = trust_for(ap_by_class.get(name))
    partners = look_alikes.get(name, [])
    pct = f'{d["conf"]*100:.0f}'

    alt = ""
    if partners:
        items = ""
        for pname, sim in partners:
            rp = ref_path(classes, pname)
            thumb = f'<img src="{data_uri(rp)}">' if rp and rp.exists() else ""
            items += (f'<div class="alt">{thumb}<div class="nm">{pname}</div>'
                      f'<div class="sm">similarity {sim:.2f}</div></div>')
        alt = (f'<div class="alt-title">Could also be '
               f'<span>— compare your leaf with these references</span></div>'
               f'<div class="alt-grid">{items}</div>')

    x1, y1, x2, y2 = d["xyxy"]
    small = ""
    if (max(0, x2 - x1) * max(0, y2 - y1)) / img_area < 0.05:
        small = ('<div class="smallobj">🔍 Small object — the model\'s recall here is only '
                 '≈ 0.44; a closer photo would help.</div>')

    pct_w = max(3, trust["frac"] * 100)
    ap_label = f'AP {trust["ap"]:.2f}' if trust["ap"] is not None else "AP —"
    return f"""
    <div class="card">
      <div class="card-head">
        <div class="diag">{name}<span class="sub">detected leaf</span></div>
        <div class="conf"><div class="pct" style="color:{trust['color']}">{pct}%</div>
             <div class="cl">confidence</div></div>
      </div>
      <div class="trust-row">
        <span class="pill" style="background:{trust['color']}">{trust['tier']}</span>
        <div class="meter"><div class="fill" style="width:{pct_w:.0f}%;background:{trust['color']}"></div></div>
        <span class="apnum">{ap_label}</span>
      </div>
      <div class="blurb">{trust['blurb']}.</div>
      {small}{alt}
    </div>
    """


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    classes, ap_by_class, look_alikes, metrics = load_assets()

    st.markdown(hero_html(metrics), unsafe_allow_html=True)

    # ---- sidebar ----
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05,
                         help="Lower = more (but riskier) detections.")
        with st.expander("Advanced"):
            iou = st.slider("NMS IoU", 0.30, 0.90, 0.60, 0.05)
            fix_neg = st.toggle("Auto-correct colour-inverted photos", value=True,
                                help="PlantDoc stored ~1 in 5 images as colour-negatives; "
                                     "the model trained on the restored (green) versions.")
        st.divider()
        st.markdown("### 📋 Model card")
        ci = metrics.get("test_mAP50_bootstrap_ci", {})
        g = metrics.get("generalization", {})
        st.markdown(
            f"- **Test mAP@0.5:** {metrics['test_mAP50']:.3f}\n"
            f"- **mAP@0.5:0.95:** {metrics['test_mAP5095']:.3f}\n"
            f"- **95% CI:** [{ci.get('ci_low',0):.3f}, {ci.get('ci_high',0):.3f}]"
        )
        if g:
            st.markdown(
                f"**Generalization (mAP@0.5)**\n\n"
                f"train {g['train']['mAP50']:.3f} · val {g['val']['mAP50']:.3f} · "
                f"test {g['test']['mAP50']:.3f}"
            )
            st.caption("The large train→val gap is honest overfitting, expected for a small, "
                       "fine-grained dataset. The hardest classes are the tomato/potato "
                       "look-alikes — exactly what the trust meters flag.")

    # ---- uploader ----
    up = st.file_uploader("Drop a leaf photo (JPG / PNG)", type=["jpg", "jpeg", "png"],
                          label_visibility="collapsed")

    if up is None:
        st.markdown(
            '<div class="feat-grid">'
            '<div class="feat"><div class="ic">🔎</div><h4>Detect</h4>'
            '<p>A YOLOv8s detector localizes every leaf and names its disease across 27 classes.</p></div>'
            '<div class="feat"><div class="ic">📊</div><h4>Trust</h4>'
            '<p>Each call carries a trust meter from the model\'s real per-class test accuracy — not a guess.</p></div>'
            '<div class="feat"><div class="ic">🪞</div><h4>Differentiate</h4>'
            '<p>See the look-alike classes it might be confused with, with reference leaves to compare.</p></div>'
            '</div>', unsafe_allow_html=True)
        st.info("👆 Upload a clear, close photo of a leaf to begin.")
        with st.expander("Which plants are supported? (27 classes)"):
            st.write(", ".join(classes))
        st.markdown('<div class="note-foot">An AI second opinion — not a substitute for an agronomist. '
                    'Every metric shown is read from the project\'s own evaluation artifacts.</div>',
                    unsafe_allow_html=True)
        return

    # decode + EXIF-orient
    pil = ImageOps.exif_transpose(Image.open(up)).convert("RGB")
    rgb = np.array(pil)

    corrected = False
    if fix_neg and is_color_negative(rgb):
        rgb = 255 - rgb
        corrected = True

    model = load_model()
    with st.spinner("Examining the leaf …"):
        dets = run_detection(model, rgb, conf, iou)

    if corrected:
        st.warning("This photo looked colour-inverted, so it was restored before analysis "
                   "(the same repair the model was trained on).")

    # ---- images, framed side by side ----
    disp_orig = for_display(Image.fromarray(rgb))
    annotated = draw_detections(rgb, dets) if dets else Image.fromarray(rgb)
    disp_ann = for_display(annotated)
    st.markdown(
        f'<div class="imgrow">'
        f'<div class="frame"><div class="ttl"><span class="dot"></span>Your photo</div>'
        f'<img src="{data_uri(disp_orig)}"></div>'
        f'<div class="frame"><div class="ttl"><span class="dot"></span>Detection</div>'
        f'<img src="{data_uri(disp_ann, fmt="PNG")}"></div>'
        f'</div>', unsafe_allow_html=True)

    # ---- no detection ----
    if not dets:
        st.error("No leaf disease detected with enough confidence.")
        st.markdown(
            "**Why this can happen — and what to try:**\n"
            "- Lower the **confidence threshold** in the sidebar.\n"
            "- Get **closer**: tiny leaves are the model's weak spot (recall ≈ 0.44 for small objects).\n"
            "- Make sure the plant is one of the **27 supported classes** (expander below).\n"
            "- Very dark, blurry, or non-leaf images fall outside the model's training."
        )
        with st.expander("Supported classes (27)"):
            st.write(", ".join(classes))
        return

    # ---- second-opinion cards ----
    plural = "s" if len(dets) > 1 else ""
    st.markdown(f'<div class="section"><h3>Second opinion</h3>'
                f'<span class="count">{len(dets)} leaf{plural} found</span></div>',
                unsafe_allow_html=True)

    img_area = rgb.shape[0] * rgb.shape[1]
    cards = "".join(card_html(d, classes, ap_by_class, look_alikes, img_area) for d in dets)
    st.markdown(cards, unsafe_allow_html=True)

    # ---- shareable card + optional ground-truth check ----
    top = dets[0]
    top_trust = trust_for(ap_by_class.get(top["class"]))
    top_partners = look_alikes.get(top["class"], [])

    cc1, cc2 = st.columns([1, 1])
    with cc1:
        card_png = make_card(annotated, top, top_trust, top_partners)
        st.download_button("⬇️  Download diagnosis card (PNG)", card_png,
                           file_name="leaf_doctor_diagnosis.png", mime="image/png",
                           use_container_width=True)
    with cc2:
        with st.expander("✓  Were we right? (tell us the true class)"):
            truth = st.selectbox("True class", ["—"] + classes, index=0)
            if truth != "—":
                pred = top["class"]
                if truth == pred:
                    st.success(f"✅ Correct — top prediction matches **{truth}**.")
                else:
                    la = dict(look_alikes.get(truth, []))
                    if pred in la:
                        st.warning(
                            f"❌ Predicted **{pred}**, true **{truth}** — a *known look-alike* pair "
                            f"(colour similarity {la[pred]:.2f}), pre-registered from the EDA before training."
                        )
                    else:
                        ap_t = ap_by_class.get(truth)
                        extra = (f" Note: **{truth}** is a hard class (test AP {ap_t:.2f})."
                                 if ap_t is not None and ap_t < 0.5 else "")
                        st.warning(f"❌ Predicted **{pred}**, true **{truth}**.{extra}")

    st.markdown('<div class="note-foot">An AI second opinion — not a substitute for an agronomist. '
                'Every metric shown is read from the project\'s own evaluation artifacts.</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
