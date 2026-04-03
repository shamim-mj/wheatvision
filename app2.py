import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="WheatVision AI", page_icon="🌾", layout="wide")

# Custom CSS for a "modern" feel
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🌾 WheatVision AI: Agricultural Analysis")
st.markdown("---")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("🛠️ Model Configuration")
    conf_threshold = st.slider("Sensitivity (Confidence)", 0.0, 1.0, 0.25, help="Lower = find more heads, Higher = more certain.")
    st.info("💡 Pro-Tip: For dense fields, try 0.25 - 0.35.")
    
    st.divider()
    st.markdown("### 📊 Dataset Stats\n- Model: YOLO11\n- Training: 100 Epochs\n- Classes: 1 (Wheat Head)")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Make sure best.pt is in the same folder as this script!
    return YOLO("best.pt")

model = load_model()

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("📤 Upload Field Imagery (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read Image
    image = Image.open(uploaded_file)
    
    # 2-Column Layout for Input vs Output
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Capture")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("AI Analysis")
        with st.spinner("Processing pixels..."):
            start_time = time.time()
            results = model.predict(source=image, conf=conf_threshold)
            end_time = time.time()
            
            # Extract Data
            count = len(results[0].boxes)
            inference_time = (end_time - start_time) * 1000 # to ms
            
            # Plot
            res_plotted = results[0].plot()
            st.image(res_plotted, use_container_width=True)

    # --- RESULTS DASHBOARD ---
    st.markdown("### 📈 Detection Results")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.metric("Total Wheat Heads", f"{count}")
    with m_col2:
        st.metric("Detection Speed", f"{inference_time:.1f} ms")
    with m_col3:
        status = "High Yield" if count > 50 else "Standard Yield"
        st.metric("Density Status", status)

    # Allow user to download the result
    st.divider()
    result_img = Image.fromarray(res_plotted)
    st.download_button("📥 Download Annotated Image", data=uploaded_file, file_name="wheat_detected.jpg")

else:
    # Welcome message when no image is uploaded
    st.info("Please upload an image from your computer to start the detection process.")
