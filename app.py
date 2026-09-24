import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
cv2.setNumThreads(0)

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from ultralytics import YOLO
import streamlit as st
import pandas as pd
import math
import io

# ─────────────────────────────────────────────────────────────────────────────
# 1. PHONE SPECS DATABASE (38 models)
# focal_mm = physical focal length in mm (NOT 35mm equivalent)
# sensor_w / sensor_h = physical sensor size in mm
# ─────────────────────────────────────────────────────────────────────────────
PHONE_SPECS = {
    "iPhone 12":                {"focal_mm": 4.2,  "sensor_w": 5.76,  "sensor_h": 4.29},
    "iPhone 13":                {"focal_mm": 5.1,  "sensor_w": 7.01,  "sensor_h": 5.26},
    "iPhone 13 Pro":            {"focal_mm": 5.7,  "sensor_w": 7.52,  "sensor_h": 5.64},
    "iPhone 14":                {"focal_mm": 5.1,  "sensor_w": 7.01,  "sensor_h": 5.26},
    "iPhone 14 Pro":            {"focal_mm": 6.86, "sensor_w": 9.0,   "sensor_h": 6.75},
    "iPhone 15":                {"focal_mm": 6.86, "sensor_w": 9.0,   "sensor_h": 6.75},
    "iPhone 15 Pro":            {"focal_mm": 6.86, "sensor_w": 9.0,   "sensor_h": 6.75},
    "iPhone 16":                {"focal_mm": 5.1,  "sensor_w": 7.01,  "sensor_h": 5.26},
    "iPhone 16 Plus":           {"focal_mm": 5.1,  "sensor_w": 7.01,  "sensor_h": 5.26},
    "iPhone 16 Pro":            {"focal_mm": 6.86, "sensor_w": 9.6,   "sensor_h": 7.2},
    "iPhone 16 Pro Max":        {"focal_mm": 6.86, "sensor_w": 9.6,   "sensor_h": 7.2},
    "iPhone 17":                {"focal_mm": 5.1,  "sensor_w": 7.01,  "sensor_h": 5.26},
    "iPhone 17 Air":            {"focal_mm": 5.1,  "sensor_w": 7.01,  "sensor_h": 5.26},
    "iPhone 17 Pro":            {"focal_mm": 6.86, "sensor_w": 9.6,   "sensor_h": 7.2},
    "iPhone 17 Pro Max":        {"focal_mm": 6.86, "sensor_w": 9.6,   "sensor_h": 7.2},
    "iPhone 18 Pro":            {"focal_mm": 6.86, "sensor_w": 9.6,   "sensor_h": 7.2},
    "iPhone 18 Pro Max":        {"focal_mm": 6.86, "sensor_w": 9.6,   "sensor_h": 7.2},
    "Samsung Galaxy S21":       {"focal_mm": 5.4,  "sensor_w": 7.2,   "sensor_h": 5.4},
    "Samsung Galaxy S22":       {"focal_mm": 6.4,  "sensor_w": 8.0,   "sensor_h": 6.0},
    "Samsung Galaxy S23":       {"focal_mm": 6.4,  "sensor_w": 8.0,   "sensor_h": 6.0},
    "Samsung Galaxy S24":       {"focal_mm": 6.4,  "sensor_w": 8.0,   "sensor_h": 6.0},
    "Samsung Galaxy S25":       {"focal_mm": 5.1,  "sensor_w": 7.01,  "sensor_h": 5.26},
    "Samsung Galaxy S25+":      {"focal_mm": 5.1,  "sensor_w": 7.01,  "sensor_h": 5.26},
    "Samsung Galaxy S25 Ultra": {"focal_mm": 6.4,  "sensor_w": 9.6,   "sensor_h": 7.2},
    "Samsung Galaxy S26":       {"focal_mm": 5.1,  "sensor_w": 7.01,  "sensor_h": 5.26},
    "Samsung Galaxy S26+":      {"focal_mm": 5.1,  "sensor_w": 7.01,  "sensor_h": 5.26},
    "Samsung Galaxy S26 Ultra": {"focal_mm": 6.4,  "sensor_w": 9.8,   "sensor_h": 7.35},
    "Samsung Galaxy A25":       {"focal_mm": 3.6,  "sensor_w": 4.8,   "sensor_h": 3.6},
    "Samsung Galaxy A35":       {"focal_mm": 4.8,  "sensor_w": 6.4,   "sensor_h": 4.8},
    "Samsung Galaxy A54":       {"focal_mm": 5.2,  "sensor_w": 6.4,   "sensor_h": 4.8},
    "Samsung Galaxy A55":       {"focal_mm": 5.1,  "sensor_w": 7.01,  "sensor_h": 5.26},
    "Google Pixel 7":           {"focal_mm": 6.81, "sensor_w": 9.0,   "sensor_h": 6.75},
    "Google Pixel 8":           {"focal_mm": 6.81, "sensor_w": 9.0,   "sensor_h": 6.75},
    "Google Pixel 8 Pro":       {"focal_mm": 6.81, "sensor_w": 9.0,   "sensor_h": 6.75},
    "Google Pixel 9":           {"focal_mm": 6.81, "sensor_w": 9.0,   "sensor_h": 6.75},
    "OnePlus 11":               {"focal_mm": 6.0,  "sensor_w": 8.0,   "sensor_h": 6.0},
    "Xiaomi 13":                {"focal_mm": 6.0,  "sensor_w": 8.0,   "sensor_h": 6.0},
    "Xiaomi 14":                {"focal_mm": 6.0,  "sensor_w": 8.0,   "sensor_h": 6.0},
    "Other / Unknown":          {"focal_mm": 4.5,  "sensor_w": 6.17,  "sensor_h": 4.55},
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXIF UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def get_exif(img: Image.Image) -> dict:
    """Dual-method EXIF extraction for maximum compatibility."""
    try:
        raw = img._getexif()
        if raw:
            return {TAGS.get(k, k): v for k, v in raw.items()}
    except Exception:
        pass
    try:
        raw = img.getexif()
        if raw:
            return {TAGS.get(k, k): v for k, v in raw.items()}
    except Exception:
        pass
    return {}

def detect_phone_from_exif(exif: dict) -> str:
    """Match EXIF Make/Model to PHONE_SPECS entry."""
    make  = str(exif.get("Make",  "")).strip().lower()
    model = str(exif.get("Model", "")).strip().lower()
    if "apple" in make or "iphone" in model:
        for key in PHONE_SPECS:
            if key.lower() in model:
                return key
        return "iPhone 17"
    if "samsung" in make or "samsung" in model:
        for key in PHONE_SPECS:
            if key.lower().replace(" ", "") in model.replace(" ", "").replace("-", ""):
                return key
        return "Samsung Galaxy S25"
    if "google" in make or "pixel" in model:
        for key in PHONE_SPECS:
            if "pixel" in key.lower() and key.lower().split()[-1] in model:
                return key
        return "Google Pixel 9"
    return "Other / Unknown"

def get_focal_from_exif(exif: dict):
    """
    Robustly extract physical focal length in mm.
    Handles float, int, tuple rational (450,100), PIL IFDRational.
    """
    fl = exif.get("FocalLength")
    if fl is None:
        return None
    try:
        if isinstance(fl, (int, float)):
            val = float(fl)
        elif isinstance(fl, tuple) and len(fl) == 2:
            val = fl[0] / fl[1]
        else:
            val = float(fl)
        if val > 200:
            val = val / 100   # rescue common rational artifact
        if val < 1.0:
            return None
        return round(val, 2)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 3. AREA CALCULATION
# area = (sensor_w * sensor_h / focal²) * height²  — resolution independent!
# ─────────────────────────────────────────────────────────────────────────────
def calculate_area(img_w_px: int, img_h_px: int,
                   height_m: float, phone_model: str,
                   focal_override=None) -> dict:
    specs = PHONE_SPECS.get(phone_model, PHONE_SPECS["Other / Unknown"])
    if focal_override and 1.0 <= focal_override <= 200.0:
        f = focal_override
        focal_source = "EXIF"
    else:
        f = specs["focal_mm"]
        focal_source = "database"

    sw = specs["sensor_w"]
    sh = specs["sensor_h"]

    ground_w_m = (sw * height_m) / f
    ground_h_m = (sh * height_m) / f
    area_m2    = ground_w_m * ground_h_m
    area_sqft  = area_m2 * 10.7639
    gsd_cm     = (sw * height_m) / (f * img_w_px) * 100

    return {
        "GSD_cm":       round(gsd_cm, 3),
        "ground_w_m":   round(ground_w_m, 3),
        "ground_h_m":   round(ground_h_m, 3),
        "area_m2":      round(area_m2, 4),
        "area_sqft":    round(area_sqft, 3),
        "focal_used":   round(f, 2),
        "focal_source": focal_source,
    }

def area_preview(height_m: float, phone_model: str,
                 focal_override=None) -> tuple:
    """
    Sidebar preview — uses actual detected phone+focal from session_state
    so it always matches the uploaded photo accurately.
    """
    specs = PHONE_SPECS.get(phone_model, PHONE_SPECS["Other / Unknown"])
    f  = focal_override if (focal_override and 1.0 <= focal_override <= 200.0) \
         else specs["focal_mm"]
    sw = specs["sensor_w"]
    sh = specs["sensor_h"]
    area_m2 = (sw * sh / f**2) * height_m**2
    return round(area_m2, 4), round(area_m2 * 10.7639, 3)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SESSION STATE — persist detected camera info across reruns
# ─────────────────────────────────────────────────────────────────────────────
if "detected_phone"  not in st.session_state:
    st.session_state.detected_phone  = "Other / Unknown"
if "detected_focal"  not in st.session_state:
    st.session_state.detected_focal  = None
if "detected_label"  not in st.session_state:
    st.session_state.detected_label  = "No photo uploaded yet"

# ─────────────────────────────────────────────────────────────────────────────
# 5. PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="WheatVision AI", page_icon="🌾", layout="centered")

st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
.stTabs [data-baseweb="tab"] {
    height: 50px; background-color: #f8f9fa; border-radius: 10px 10px 0 0;
    padding: 10px 25px; color: #444; border: 1px solid #e0e0e0; transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background-color: #004aad !important; color: white !important;
    font-weight: bold; border: 1px solid #004aad;
    box-shadow: 0 4px 10px rgba(0,74,173,0.2); transform: translateY(-2px);
}
.stTabs [data-baseweb="tab"]:hover { background-color: #eef2f6; color: #004aad; }
.stTabs [data-baseweb="tab-highlight"] { background-color: transparent !important; }
.header-box {
    background-color: rgba(255,255,255,0.8); backdrop-filter: blur(10px);
    padding: 0; border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    text-align: center; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.3);
}
@media only screen and (max-width: 600px) { .header-box { padding: 5px; } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1 style='text-align:center;color:#004aad;font-size:2.3rem;margin-bottom:0'>🌾 WheatVision AI</h1>
    <p style='text-align:center;color:#666;font-size:1.1rem;margin-top:-10px'>
        Wheat Head Detection & Yield Assessment
    </p>
    <hr style='margin-top:1px;margin-bottom:1px'>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# 7. SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🛠️ Settings")
    conf_threshold = st.slider("Sensitivity", 0.0, 1.0, 0.16)

    st.divider()
    st.subheader("📐 Area Measurement")

    st.markdown("""
    <div style='background:#fff8e1;padding:10px;border-radius:8px;
                border-left:3px solid #f59e0b;font-size:0.85em;margin-bottom:8px'>
        <b>📏 Height Guide:</b><br>
        Height = distance from lens to wheat heads.<br>
        • 0.3 m = 1 ft &nbsp;→&nbsp; ~1 ft² area<br>
        • 0.5 m = 1.6 ft → ~3 ft² area<br>
        • 1.0 m = 3.3 ft → ~10 ft² area
    </div>
    """, unsafe_allow_html=True)

    enable_area = st.toggle("Enable Area Calculation", value=True)

    if enable_area:
        height_m = st.slider(
            "📏 Phone Height Above Crop (m)",
            min_value=0.1, max_value=2.0, value=0.7, step=0.05,
            help="0.3m ≈ 1 ft | 0.5m ≈ 1.6 ft | 1.0m ≈ 3.3 ft"
        )

        st.caption("📱 Phone model — auto-detected or select manually:")
        manual_phone = st.selectbox(
            "Phone Model",
            ["🔍 Auto-detect from photo"] + list(PHONE_SPECS.keys()),
            index=0,
            label_visibility="collapsed"
        )
        auto_detect = (manual_phone == "🔍 Auto-detect from photo")
        if auto_detect:
            manual_phone = None

        # ── SMART PREVIEW: uses actual detected camera after upload ──
        if auto_detect:
            preview_phone = st.session_state.detected_phone
            preview_focal = st.session_state.detected_focal
        else:
            preview_phone = manual_phone
            preview_focal = None

        prev_m2, prev_sqft = area_preview(height_m, preview_phone, preview_focal)

        # Show which camera the preview is based on
        cam_label = st.session_state.detected_label if auto_detect \
                    else f"Manual: {manual_phone}"

        # REPLACE the st.info() block with this:
        st.markdown(f"""
        <div style='background:#e8f0fe;padding:12px;border-radius:8px;
                    font-size:0.9em;margin-top:5px'>
            <div style='font-size:1.2em;font-weight:bold;color:#004aad'>
                {prev_m2} m² &nbsp;({prev_sqft} ft²)
            </div>
            <div style='color:#555;margin-top:4px;font-size:0.85em'>
                per photo
            </div>
            <hr style='margin:8px 0;border-color:#c5d5f5'>
            <div style='color:#444;font-size:0.82em'>
                📷 {cam_label}
            </div>
            <div style='color:#444;font-size:0.82em;margin-top:4px'>
                📏 {height_m} m &nbsp;({height_m * 3.281:.1f} ft) height
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 8. TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Single Analysis", "📂 Batch Processing", "📧 Contact & Info"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown("""
    <div style='background:#f0f7ff;padding:15px;border-radius:10px;
                border-left:4px solid #004aad;margin-bottom:15px'>
        <b>📱 How to Take a Good Field Photo</b>
        <ol style='margin:8px 0;padding-left:20px;color:#333'>
            <li>Open your phone's <b>native Camera app</b></li>
            <li>Hold phone <b>straight above</b> wheat heads at your chosen height</li>
            <li>Point camera <b>straight down (90°)</b> — use phone level if needed</li>
            <li>Take the photo, come back here and upload below ⬇️</li>
        </ol>
        <small style='color:#666'>💡 Native camera gives better quality.
        Set height in sidebar to match how high you held the phone.</small>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload Field Photo (from Camera Roll or Files)",
        type=["jpg", "png", "jpeg"],
        key="single_up",
        help="iPhone: tap Browse → Photos | Android: tap Files → Gallery"
    )

    with st.expander("📷 Or use in-app camera (optional / limited)"):
        st.warning("⚠️ May default to **front (selfie) camera** on some phones. "
                   "Switch to rear manually if needed.")
        camera_file = st.camera_input("", label_visibility="collapsed", key="wheat_cam")

    input_source = uploaded_file if uploaded_file else camera_file

    if input_source:
        img = Image.open(input_source)
        img_w_px, img_h_px = img.size

        # EXIF detection
        exif           = get_exif(img)
        detected_phone = detect_phone_from_exif(exif)
        focal_exif     = get_focal_from_exif(exif)
        has_make       = bool(exif.get("Make") or exif.get("Model"))

        # ── Update session state so sidebar preview matches this photo ──
        if auto_detect:
            st.session_state.detected_phone = detected_phone
            st.session_state.detected_focal = focal_exif
            make_str  = exif.get("Make",  "")
            model_str = exif.get("Model", "")
            st.session_state.detected_label = \
                f"{make_str} {model_str} → {detected_phone}".strip()

        # Resolve final phone & focal
        if enable_area:
            phone_used = detected_phone if auto_detect else (manual_phone or "Other / Unknown")
            focal_used = focal_exif if auto_detect else None
            area_info  = calculate_area(img_w_px, img_h_px, height_m, phone_used, focal_used)

        # ── EXIF FEEDBACK ─────────────────────────
        if enable_area:
            if has_make and focal_exif:
                st.success(
                    f"📷 Detected: **{exif.get('Make','')} {exif.get('Model','')}** "
                    f"→ **{phone_used}** | Focal: **{focal_exif}mm** (EXIF ✅)"
                )
            elif has_make:
                st.info(
                    f"📷 Detected: **{exif.get('Make','')} {exif.get('Model','')}** "
                    f"→ **{phone_used}** | Focal from database: **{area_info['focal_used']}mm**"
                )
            else:
                st.warning(
                    "⚠️ **Camera not detected** — browser removed photo metadata.\n\n"
                    "Please select your phone/camera model in the sidebar."
                )
                # ← ADD THIS LINE right here, always shows after any EXIF result:
            st.info("💡 If the detected phone model is incorrect, please select the correct one from the sidebar dropdown.")

        # ── RUN AI ────────────────────────────────
        with st.spinner("🤖 AI is counting wheat heads..."):
            results     = model.predict(source=img, conf=conf_threshold, max_det=2000)
            res_plotted = results[0].plot(labels=False, line_width=2, probs=False, boxes=True)
            res_rgb     = Image.fromarray(res_plotted[..., ::-1])
            count       = len(results[0].boxes)

        # ── IMAGES — use_container_width (fixes deprecation warning) ──
        st.subheader("📊 Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original", use_container_width=True)
        with col2:
            st.image(res_rgb, caption=f"Detected: {count} heads", use_container_width=True)

        st.metric("🌾 Total Wheat Heads Detected", f"{count}")

        # ── AREA & DENSITY ────────────────────────
        if enable_area:
            st.divider()
            st.subheader("📐 Area & Density")
            st.caption(
                f"📷 **{phone_used}** | "
                f"Focal: **{area_info['focal_used']}mm** ({area_info['focal_source']}) | "
                f"Height: **{height_m}m** ({height_m * 3.281:.1f} ft)"
            )
            st.info("Please select the correct phone model if auto detection is incorrect")

            a1, a2, a3 = st.columns(3)

            a1.metric("📐 Area Covered", f"{area_info['area_m2']} m²")
            a1.caption(f"≈ {area_info['area_sqft']} ft²")

            a2.metric("🌾 Heads per m²",
                    f"{round(count / area_info['area_m2'], 1)}"
                    if area_info['area_m2'] > 0 else "—")

            a3.metric("📏 GSD", f"{area_info['GSD_cm']} cm/px")

            heads_per_m2 = count / area_info['area_m2'] if area_info['area_m2'] > 0 else 0
            st.info(
                f"📊 Estimated **{int(heads_per_m2 * 10000):,} heads/ha** "
                f"({int(heads_per_m2 * 4047):,} heads/acre)"
            )

        # ── YIELD PREDICTOR ───────────────────────
        st.divider()
        with st.expander("📈 Advanced Yield Predictor (Optional)"):
            default_sqft = area_info['area_sqft'] if enable_area else 1.0
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                avg_grains  = st.number_input("Avg. Grains per Head", min_value=1, value=35)
                sample_area = st.number_input("Sample Area (sq. ft.)",
                                              min_value=0.01, value=float(default_sqft),
                                              help="Auto-filled from area above ✅" if enable_area else "")
            with col_c2:
                tgw = st.number_input("1000-Grain Weight (grams)", min_value=20.0, value=35.0)

            if st.button("🧮 Calculate Predicted Yield"):
                total_wt_grams   = (count * avg_grains * tgw) / 1000
                lbs_per_acre     = (total_wt_grams / sample_area) * 96.047
                bushels_per_acre = lbs_per_acre / 60
                st.divider()
                st.subheader(f"🌾 Estimated Yield: {bushels_per_acre:.2f} bu/ac")
                st.info(f"Based on **{count}** heads in **{sample_area:.3f} ft²** "
                        f"| **{avg_grains}** grains/head @ **{tgw}g** TGW")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    uploaded_files = st.file_uploader(
        "📂 Upload Multiple Photos",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch"
    )

    if uploaded_files:
        batch_results = []
        progress_bar  = st.progress(0)

        for i, file in enumerate(uploaded_files):
            img_b        = Image.open(file)
            img_w, img_h = img_b.size
            exif_b       = get_exif(img_b)
            focal_b      = get_focal_from_exif(exif_b)
            phone_b      = detect_phone_from_exif(exif_b) if auto_detect \
                           else (manual_phone or "Other / Unknown")

            results_b = model.predict(source=img_b, conf=conf_threshold)
            count_b   = len(results_b[0].boxes)

            row = {
                "No.":         i + 1,
                "Photo Name":  file.name,
                "Camera":      phone_b,
                "Focal (mm)":  focal_b if focal_b else "—",
                "Wheat Heads": count_b,
            }

            if enable_area:
                area_b = calculate_area(img_w, img_h, height_m, phone_b, focal_b)
                row["Height (m)"]  = height_m
                row["Area (m²)"]   = area_b["area_m2"]
                row["Area (ft²)"]  = area_b["area_sqft"]
                row["Heads/m²"]    = round(count_b / area_b["area_m2"], 1) \
                                     if area_b["area_m2"] > 0 else 0
                row["GSD (cm/px)"] = area_b["GSD_cm"]

            batch_results.append(row)
            progress_bar.progress((i + 1) / len(uploaded_files))

        df = pd.DataFrame(batch_results)

        if enable_area and "Heads/m²" in df.columns:
            c1, c2, c3 = st.columns(3)
            c1.metric("📸 Images",       len(df))
            c2.metric("🌾 Avg Heads/m²", f"{df['Heads/m²'].mean():.1f}")
            c3.metric("📐 Total Area",   f"{df['Area (m²)'].sum():.3f} m²")

        st.success(f"✅ Processed {len(uploaded_files)} images successfully!")
        st.dataframe(df, use_container_width=True)
        st.divider()
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results as CSV", csv,
                           "wheatvision_report.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONTACT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([0.1, 2.8, 0.1])
    with center_col:
        st.markdown("<p style='text-align:center;color:#666'>Have questions about WheatVision AI "
                    "or my research? Send a message below.</p>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h4 style='background-color:#0033A0;text-align:center;color:white;"
                        "padding:10px;border-radius:5px'>Mohammad Jan Shamim</h4>",
                        unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align:center;margin-bottom:20px'>
                <p style='margin:10px 0'><b>Crop Physiologist / Agronomist /
                Grain Crops Extension Associate</b></p>
                <p style='font-size:0.9em;color:#555'>Specializing in eco-physiological responses,
                Data Analytics & Decision Support Tools</p>
                <p style='font-size:0.9em;color:#555;font-weight:bold'>University of Kentucky</p>
            </div>
            """, unsafe_allow_html=True)
            contact_form = """
            <form action="https://formsubmit.co/shamim.one@outlook.com" method="POST">
                <input type="hidden" name="_captcha" value="false">
                <div style="margin-bottom:10px">
                    <input type="text" name="name" placeholder="Your Name"
                           style="width:100%;padding:10px;border-radius:5px;border:1px solid #ccc" required>
                </div>
                <div style="margin-bottom:10px">
                    <input type="email" name="email" placeholder="Email Address"
                           style="width:100%;padding:10px;border-radius:5px;border:1px solid #ccc" required>
                </div>
                <div style="margin-bottom:10px">
                    <textarea name="message" placeholder="How can I help with your wheat analysis?"
                              style="width:100%;padding:10px;border-radius:5px;
                                     border:1px solid #ccc;height:100px"></textarea>
                </div>
                <button type="submit"
                        style="background-color:#0033A0;color:white;border:none;padding:12px 20px;
                               border-radius:5px;cursor:pointer;width:100%;font-weight:bold">
                    🚀 Send Message
                </button>
            </form>"""
            st.markdown(contact_form, unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center;margin-top:20px'>
            <a href="mailto:shamim.one@outlook.com"
               style="text-decoration:none;color:#0033A0;font-weight:bold">📧 Email</a> |
            <a href="https://github.com/shamim-mj"
               style="text-decoration:none;color:#0033A0;font-weight:bold">💻 GitHub</a>
        </div>""", unsafe_allow_html=True)
