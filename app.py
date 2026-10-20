import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
cv2.setNumThreads(0)

import math
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from ultralytics import YOLO
import streamlit as st

# =============================================================
# 1. PAGE CONFIG & STYLING
# =============================================================
st.set_page_config(page_title="WheatVision AI", page_icon="🌾", layout="centered")

st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
.stTabs [data-baseweb="tab"] {
    height: 50px; background-color: #f8f9fa; border-radius: 10px 10px 0 0;
    padding: 10px 25px; color: #444; border: 1px solid #e0e0e0; transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background-color: #004aad !important; color: white !important; font-weight: bold;
    border: 1px solid #004aad; box-shadow: 0 4px 10px rgba(0,74,173,0.2); transform: translateY(-2px);
}
.stTabs [data-baseweb="tab"]:hover { background-color: #eef2f6; color: #004aad; }
.stTabs [data-baseweb="tab-highlight"] { background-color: transparent !important; }

.header-box {
    background-color: rgba(255,255,255,0.8); backdrop-filter: blur(10px); padding: 0;
    border-radius: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); text-align: center;
    margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.3);
}
.tip-card {
    background: #eef4ff; border-left: 5px solid #004aad; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 12px; font-size: 0.92rem; color: #222; line-height: 1.6;
}
.tip-title { color: #004aad; font-size: 1rem; font-weight: bold; }
@media only screen and (max-width: 1000px) {
    /* Main page container: less side and top padding */
    [data-testid="stMainBlockContainer"], .block-container {
        padding-left: 0.1rem !important;
        padding-right: 0.1rem !important;
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
    }

    /* Header: smaller and tighter */
    .header-box { padding: 4px 6px; margin-bottom: 10px; }
    .header-box h1 { font-size: 1.7rem !important; }
    .header-box p  { font-size: 0.95rem !important; margin-top: -6px !important; }

    /* Tabs: smaller so all three fit */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px; padding: 6px 10px; font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab"] p { font-size: 0.85rem !important; }

    /* Tips card: tighter */
    .tip-card { font-size: 0.85rem; padding: 8px 10px; line-height: 1.45; }
}
</style>
<div class="header-box">
    <h1 style='text-align:center; color:#004aad; font-size:2.3rem; margin-bottom:0;'>🌾 WheatVision AI</h1>
    <p style='text-align:center; color:#666; font-size:1.1rem; margin-top:-10px;'>
        Wheat Head Detection & Yield Assessment</p>
    <hr style='margin-top:1px; margin-bottom:1px;'>
</div>
""", unsafe_allow_html=True)


# =============================================================
# 2. MODEL & CONSTANTS
# =============================================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

MAX_DET = 2000
DIAG_35MM = 43.27
M2_TO_FT2 = 10.7639
KGHA_TO_BUAC = 0.014870

lens_presets = {
    "Main camera 1x (~26 mm)": 26.0,
    "Ultra-wide 0.5x (~13 mm)": 13.0,
    "Zoom 2x (~52 mm)": 52.0,
}


# =============================================================
# 3. HELPERS
# =============================================================
def load_image(file):
    img = Image.open(file)
    f_mm, f35_mm = None, None
    try:
        exif = img.getexif()
        sub = exif.get_ifd(0x8769)
        f_raw = sub.get(37386) or exif.get(37386)
        f35_raw = sub.get(41989) or exif.get(41989)
        f_mm = float(f_raw) if f_raw else None
        f35_mm = float(f35_raw) if f35_raw else None
    except Exception:
        pass
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img, f_mm, f35_mm


def footprint_from_height(height_cm, f35_mm, w_px, h_px):
    diag_cm = height_cm * DIAG_35MM / f35_mm
    d_px = math.hypot(w_px, h_px)
    width_cm = diag_cm * w_px / d_px
    length_cm = diag_cm * h_px / d_px
    return (width_cm * length_cm) / 10_000, width_cm, length_cm


def resolve_area(img, f35_exif):
    if area_method == "Manual area (quadrat / frame)":
        return manual_area_ft2 / M2_TO_FT2, None, "Manual area"
    if lens_choice == "Auto (photo EXIF)":
        if f35_exif:
            f35, note = f35_exif, "EXIF"
        else:
            f35, note = 26.0, "No EXIF → assumed 26 mm (1x)"
    else:
        f35, note = lens_presets[lens_choice], lens_choice
    area_m2, _, _ = footprint_from_height(camera_height_cm, f35, img.width, img.height)
    return area_m2, f35, note


# def run_model(img):
#     return model.predict(source=img, conf=conf_threshold, max_det=MAX_DET, verbose=False)
BASE_IMGSZ = 640     # the size the model works best at
REF_WIDTH_CM = 70    # ground width (long side) where 640 works best (≈ 50 cm height, 1x lens)
MAX_IMGSZ = 2080     # memory limit; Streamlit Cloud may need 1600

def auto_imgsz(img, f35_exif):
    """Pick imgsz so wheat heads are about the same pixel size as at 50 cm / 640."""
    if area_method != "Fixed camera height":
        return BASE_IMGSZ                                   # quadrat/cropped photos → default
    if lens_choice == "Auto (photo EXIF)" and f35_exif:
        f35 = f35_exif
    else:
        f35 = lens_presets.get(lens_choice, 26.0)
    _, w_cm, l_cm = footprint_from_height(camera_height_cm, f35, img.width, img.height)
    size = BASE_IMGSZ * max(w_cm, l_cm) / REF_WIDTH_CM
    size = min(max(size, BASE_IMGSZ), MAX_IMGSZ, max(img.width, img.height))  # never upscale past the photo
    return int(size // 32 * 32)                             # YOLO needs a multiple of 32


def run_model(img, imgsz=BASE_IMGSZ):
    return model.predict(source=img, conf=conf_threshold, max_det=MAX_DET,
                         imgsz=imgsz, verbose=False)


# =============================================================
# 4. SIDEBAR  (settings + FULL PHOTO GUIDE)
# =============================================================
with st.sidebar:
    st.header("🛠️ Settings")
    conf_threshold = st.slider("Sensitivity", 0.0, 1.0, 0.16)
    st.divider()

    st.subheader("📐 Sampled Area")
    area_method = st.radio("How should area be determined?",
                           ["Fixed camera height", "Manual area (quadrat / frame)"])

    # safe defaults so nothing breaks when switching methods
    manual_area_ft2 = 1.0
    camera_height_cm = 50.0
    lens_choice = "Main camera 1x (~26 mm)"

    if area_method == "Fixed camera height":
        camera_height_cm = st.number_input(
            "Lens height above the HEADS (cm)", min_value=10.0, max_value=300.0, value=50.0, step=5.0,
            help="Measure from the phone lens to the top of the wheat heads, not to the soil.")
        lens_choice = st.selectbox("Camera lens",
                                   ["Auto (photo EXIF)"] + list(lens_presets.keys()) + ["Custom"])
        if lens_choice == "Custom":
            lens_presets["Custom"] = st.number_input("35 mm-equivalent focal length (mm)", 10.0, 200.0, 26.0)
    else:
        manual_area_ft2 = st.number_input("Area in photo (sq. ft.)", min_value=0.1, value=1.0, step=0.1)

    st.divider()
    use_camera = st.toggle("📸 Enable Field Camera", value=False)
    if use_camera:
        st.caption("⚠️ In-app camera = lower resolution & no lens data. "
                "For best accuracy, use your phone's native camera and upload.")

    # ---------- FULL PHOTO GUIDE (sidebar) ----------
    st.divider()
    st.subheader("📖 Photo Guide")
    with st.expander("How to take the best photo", expanded=False):
        st.markdown("""
**🧭 Position**
- Phone **flat & level**, pointing **straight down**.
- Turn on the camera **grid / level**.

**📏 Height** *(most important)*
- Measure lens → **top of the heads**, not the soil.
- **Same height** every photo — mark a stick or string.

**🔍 Lens**
- Use your phone's native camera app, not the in-app camera — it keeps full resolution and lens data.
- Use the **main 1x** camera. Using 0.5x/2x? Select it above.
- No pinch-zoom, crop, filters or portrait mode.
- Upload the **original** file — WhatsApp/text/screenshots lose lens data.

**☀️ Light**
- Overcast, or **9–11 AM / 3–5 PM**.
- No harsh sun or **your own shadow** in the frame. Area fully shaed in the frame is fine.

**🌾 Crop stage**
- Heads fully out: **Feekes 10.5 – 11.2**.
- Avoid dew, wind, lodged spots, field edges & wheel tracks.

**🔁 Sampling**
- **5–10 photos per field** in a **W / zig-zag**.
- Use **Batch Processing** and download the CSV.

💡 **Check once:** shoot a **1 ft² quadrat** at your height — the frame should read ≈ 1 ft².
""")


# ---------- SHORT TIPS CARD (shown above the uploaders) ----------
def show_quick_tips():
    if area_method == "Fixed camera height":
        f35 = lens_presets.get(lens_choice, 26.0)
        _, w_cm, l_cm = footprint_from_height(camera_height_cm, f35, 4032, 3024)
        steps = ("1️⃣ Take photos with your phone's <b>native camera app</b>, then upload them here "
                 "(better quality + lens data for accurate area).<br>"
                 f"2️⃣ Phone <b>flat, pointing straight down</b>, lens <b>{camera_height_cm:.0f} cm above the heads</b> (not the soil).<br>"
                 "3️⃣ <b>Main 1x camera</b> — no zoom, crop or filters.<br>"
                 "4️⃣ Soft light, no half frame shadows, sharp focus. Upload the <b>original</b> photo.")
        cover = f"One photo ≈ <b>{w_cm:.0f} × {l_cm:.0f} cm</b> (<b>{w_cm * l_cm / 929.03:.1f} ft²</b>)."
    else:
        steps = ("1️⃣ Take photos with your phone's <b>native camera app</b>, then upload them here.<br>"
                 "2️⃣ Place your <b>quadrat / frame</b> on the canopy and shoot <b>straight down</b>.<br>"
                 "3️⃣ <b>Crop to the inside of the frame</b> before uploading.<br>"
                 "4️⃣ Soft light, no half frame shadows, sharp focus.")
        cover = f"Each photo counted as <b>{manual_area_ft2:.2f} ft²</b>."

    st.markdown(f"""<div class="tip-card">
<span class="tip-title">📸 Quick Photo Tips</span><br>
{steps}<br>
<span style="color:#555;">📐 {cover} &nbsp;·&nbsp; Full guide in the sidebar ⬅️</span>
</div>""", unsafe_allow_html=True)


# =============================================================
# 5. TABS
# =============================================================
tab1, tab2, tab3 = st.tabs(["🎯 Single Image", "📂 Batch", "📧 Contact"])

# ---------------- TAB 1: SINGLE ----------------
with tab1:
    show_quick_tips()
    uploaded_file = st.file_uploader("📤 Upload Field Photo", type=["jpg", "png", "jpeg"], key="single_up")
    camera_file = st.camera_input("📷 Take a Field Photo", key="wheat_cam") if use_camera else None
    input_source = uploaded_file if uploaded_file else camera_file

    if input_source:
        img, f_mm, f35_exif = load_image(input_source)
        imgsz = auto_imgsz(img, f35_exif)
        with st.spinner("🤖 AI is counting..."):
            results = run_model(img, imgsz)
            res_plotted = results[0].plot(labels=False, line_width=2, probs=False, boxes=True)
            res_plotted_rgb = Image.fromarray(res_plotted[..., ::-1])
            count = len(results[0].boxes)

        st.subheader("📊 Analysis Results")
        c1, c2 = st.columns(2)
        with c1:
            st.image(img, caption="Original")
        with c2:
            st.image(res_plotted_rgb, caption=f"Detected: {count} heads")

        area_m2, f35_used, note = resolve_area(img, f35_exif)
        heads_m2 = count / area_m2 if area_m2 > 0 else float("nan")

        m1, m2, m3 = st.columns(3)
        m1.metric("🌾 Heads", f"{count}")
        m2.metric("📐 Area", f"{area_m2 * M2_TO_FT2:.2f} ft²", f"{area_m2:.3f} m²", delta_color="off")
        m3.metric("🌾 Heads / ft²", f"{heads_m2 / M2_TO_FT2:.1f}", f"{heads_m2:.0f} /m²", delta_color="off")

        if f35_used:
            st.caption(f"Lens: {f35_used:.0f} mm (35 mm-eq.) – source: {note}"
                       + (f" | true focal length {f_mm:.2f} mm" if f_mm else "") + (f" | Model input size: {imgsz} px"))
        if area_method == "Fixed camera height" and lens_choice == "Auto (photo EXIF)" and not f35_exif:
            st.warning("This photo has no lens EXIF (common for in-app camera photos). "
                       "Select your lens in the sidebar for an accurate area.")

        with st.expander("📈 Advanced Yield Predictor (Optional)"):
            y1, y2 = st.columns(2)
            avg_grains = y1.number_input("Avg. Grains per Head", min_value=1, value=35)
            tgw = y2.number_input("1000-Grain Weight (g)", min_value=10.0, value=35.0)

            grain_g_m2 = heads_m2 * avg_grains * tgw / 1000
            kg_ha = grain_g_m2 * 10
            bu_ac = kg_ha * KGHA_TO_BUAC

            st.divider()
            st.subheader(f"🧮 Estimated Yield: {bu_ac:.1f} bu/ac")
            st.info(f"{count} heads in {area_m2 * M2_TO_FT2:.2f} ft² "
                    f"({heads_m2:.0f} heads/m²) × {avg_grains} grains × {tgw:.0f} g/1000 "
                    f"= {kg_ha:,.0f} kg/ha")

# ---------------- TAB 2: BATCH ----------------
with tab2:
    show_quick_tips()
    uploaded_files = st.file_uploader("📂 Upload Multiple Photos", type=["jpg", "jpeg", "png"],
                                      accept_multiple_files=True, key="batch")
    if uploaded_files:
        rows = []
        progress_bar = st.progress(0)
        for i, file in enumerate(uploaded_files):
            img, f_mm, f35_exif = load_image(file)
            imgsz = auto_imgsz(img, f35_exif)
            count = len(run_model(img, imgsz)[0].boxes)
            area_m2, f35_used, note = resolve_area(img, f35_exif)
            rows.append({
                "No.": i + 1,
                "Photo Name": file.name,
                "Wheat Detected": count,
                "Focal (mm)": f_mm,
                "Focal 35mm-eq (mm)": f35_used,
                "Area (ft²)": round(area_m2 * M2_TO_FT2, 3),
                "Heads/ft²": round(count / (area_m2 * M2_TO_FT2), 1),
                "Heads/m²": round(count / area_m2, 0),
                "Area Source": note,
            })
            progress_bar.progress((i + 1) / len(uploaded_files))

        df = pd.DataFrame(rows)
        for col in ["Focal (mm)", "Focal 35mm-eq (mm)", "Area (ft²)", "Heads/ft²", "Heads/m²"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        st.success(f"✅ Processed {len(uploaded_files)} images successfully!")
        st.dataframe(df, hide_index=True)
        st.divider()
        st.download_button(
            label="📥 Download Results as CSV (Excel Compatible)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="wheat_detection_report.csv",
            mime="text/csv",
        )

# ---------------- TAB 3: CONTACT ----------------
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([0.1, 2.8, 0.1])
    with center_col:
        st.markdown("<p style='text-align:center; color:#666;'>Have questions about WheatVision AI or my research? "
                    "Send a message below.</p>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h4 style='background-color:#0033A0; text-align:center; color:white; padding:10px; "
                        "border-radius:5px;'>Mohammad Shamim</h4>", unsafe_allow_html=True)
            st.markdown("""
                <div style='text-align:center; margin-bottom:20px;'>
                    <p style='margin:10px 0;'><b>Crop Physiologist / Agronomist / Grain Crops Extension Associate</b></p>
                    <p style='font-size:0.9em; color:#555;'>Specializing in eco-physiological responses, Data Analytics, & Decision Support Tools</p>
                    <p style='font-size:0.9em; color:#555; font-weight:bold;'>University of Kentucky</p>
                </div>""", unsafe_allow_html=True)
            st.markdown("""
            <form action="https://formsubmit.co/shamim.one@outlook.com" method="POST">
                <input type="hidden" name="_captcha" value="false">
                <div style="margin-bottom:10px;"><input type="text" name="name" placeholder="Your Name"
                    style="width:100%; padding:10px; border-radius:5px; border:1px solid #ccc;" required></div>
                <div style="margin-bottom:10px;"><input type="email" name="email" placeholder="Email Address"
                    style="width:100%; padding:10px; border-radius:5px; border:1px solid #ccc;" required></div>
                <div style="margin-bottom:10px;"><textarea name="message" placeholder="How can I help you with your wheat analysis?"
                    style="width:100%; padding:10px; border-radius:5px; border:1px solid #ccc; height:100px;"></textarea></div>
                <button type="submit" style="background-color:#0033A0; color:white; border:none; padding:12px 20px;
                    border-radius:5px; cursor:pointer; width:100%; font-weight:bold;">🚀 Send Message</button>
            </form>""", unsafe_allow_html=True)
        st.markdown("""
            <div style='text-align:center; margin-top:20px;'>
                <a href="mailto:shamim.one@outlook.com" style="text-decoration:none; color:#0033A0; font-weight:bold;">📧 Email</a> |
                <a href="https://github.com/shamim-mj" style="text-decoration:none; color:#0033A0; font-weight:bold;">💻 GitHub</a>
            </div>""", unsafe_allow_html=True)
