import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import pandas as pd
import io  # Required for the download button to work correctly

# --- 1. PAGE CONFIG & BLUE TAB STYLING ---
st.set_page_config(page_title="WheatVision AI", page_icon="🌾", layout="centered")

st.markdown("""
    <style>
            
    /* 1. Main Tab Container Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    /* 2. Individual Tab Styling (Inactive) */
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f8f9fa; /* Light Gray */
        border-radius: 10px 10px 0px 0px; /* Rounded Top */
        padding: 10px 25px;
        color: #444;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }

    /* 3. Selected Tab Styling (Active) */
    .stTabs [aria-selected="true"] {
        background-color: #004aad !important; /* Your Blue */
        color: white !important;
        font-weight: bold;
        border: 1px solid #004aad;
        box-shadow: 0px 4px 10px rgba(0, 74, 173, 0.2); /* Soft Blue Glow */
        transform: translateY(-2px); /* Slight lift effect */
    }

    /* 4. Hover Effect */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #eef2f6;
        color: #004aad;
    }

    /* 5. Clean up the bottom line */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- RESPONSIVE CENTERED HEADER ---
st.markdown("""
    <style>
    /* Responsive Header Container */
    .header-box {
        background-color: rgba(255, 255, 255, 0.8); /* Light beautiful background */
        backdrop-filter: blur(10px);
        padding: 0px;
        border-radius: 10px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Desktop Font Size */
    .header-title {
        color: #004aad;
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0;
    }

    /* MOBILE RESPONSIVENESS: Shrink text for phone screens */
    @media only screen and (max-width: 600px) {
        .header-title {
            font-size: 2.0rem !important; /* Smaller for mobile */
        }
        .header-box {
            padding: 5px; /* Less padding on small screens */
        }
    }
    </style>
    <div class="header-box">
            <h1 style='text-align: center; color: #004aad; font-size: 2.5rem; margin-bottom: 0px;'>
                🌾 WheatVision
            </h1>
            <p style='text-align: center; color: #666; font-size: 1.1rem; margin-top: -10px;'>
                Precision Wheat Head Detection & Yield Assessment
             </p>
            <hr style='margin-top: 1px; margin-bottom: 1px;'>
    </div>
    """, unsafe_allow_html=True)



@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- 3. SIDEBAR CONFIG ---
with st.sidebar:
    st.header("🛠️ Settings")
    conf_threshold = st.slider("Sensitivity", 0.0, 1.0, 0.16)
    st.divider()
    use_camera = st.toggle("📸 Enable Field Camera", value=False)

# --- 4. THE TABS ---
tab1, tab2, tab3 = st.tabs(["🎯 Single Analysis", "📂 Batch Processing", "📧 Contact & Info"])

# --- TAB 1: SINGLE ANALYSIS ---
with tab1:
    # Stacked Input: Upload first, Camera underneath
    uploaded_file = st.file_uploader("📤 Upload Field Photo", type=["jpg", "png", "jpeg"], key="single_up")
    
    camera_file = None
    if use_camera:
        camera_file = st.camera_input("📷 Take a Field Photo", label_visibility="collapsed")

    # Determine Source
    input_source = uploaded_file if uploaded_file else camera_file

    if input_source:
        img = Image.open(input_source)
        
        # Run AI
        with st.spinner("🤖 AI is counting..."):
            results = model.predict(source=img, conf=conf_threshold)
            # Clean Plot: No labels, thin lines
            res_plotted = results[0].plot(labels=False, line_width=2, probs=False, boxes=True)
            res_plotted_rgb = Image.fromarray(res_plotted[..., ::-1])
            count = len(results[0].boxes)

        # --- BEAUTIFUL RESULT CONTAINER ---
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.subheader("📊 Analysis Results")
        
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(img, caption="Original", use_column_width=True)
        with col_img2:
            st.image(res_plotted_rgb, caption=f"Detected: {count} heads", use_column_width=True)
        
        # Results Summary with Icons
        #c1, c2, c3 = st.columns(3)
        #c1.metric("🌾 Total Count", f"{count}")
        #c2.metric("⏱️ Process Time", f"{(time.time() - time.time()):.2f}s") # Dummy for demo
        #c3.success(f"✅ Yield found: {count} heads")
        
        st.markdown('</div>', unsafe_allow_html=True)

            # --- BEAUTIFUL RESULT CONTAINER ---
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.subheader("📊 Analysis Results")

        # ... (Your existing Image Display Columns here) ...

        # 1. Basic Metric
        st.metric("🌾 Total Wheat Heads Detected", f"{count}")

        # 2. OPTIONAL YIELD PREDICTOR
        with st.expander("📈 Advanced Yield Predictor (Optional)"):
            st.write("Enter field data to estimate total yield:")
            
            col_calc1, col_calc2 = st.columns(2)
            with col_calc1:
                avg_grains = st.number_input("Avg. Grains per Head", min_value=1, value=35)
                sample_area = st.number_input("Sample Area (sq. ft.)", min_value=0.1, value=1.0)
            
            with col_calc2:
                # Standard test weight (approx 60 lbs per bushel)
                thousand_grain_weight = st.number_input("1000-Grain Weight (grams)", min_value=20.0, value=35.0)

            # Calculation logic
            if st.button("Calculate Predicted Yield"):
                # Math: (Heads * Grains * Weight) / Area -> scaled to Acre
                total_grains = count * avg_grains
                total_weight_grams = (total_grains * thousand_grain_weight) / 1000
                
                # Convert to Bushels per Acre (approximate formula)
                # Grams per sq ft to lbs per acre
                lbs_per_acre = (total_weight_grams / sample_area) * 96.047 
                bushels_per_acre = lbs_per_acre / 60
                
                st.divider()
                st.subheader(f"🧮 Estimated Yield: {bushels_per_acre:.2f} bu/ac")
                st.info(f"Based on {count} heads in {sample_area} sq.ft. with {avg_grains} grains/head.")

st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: BATCH PROCESSING ---
with tab2:
    uploaded_files = st.file_uploader("📂 Upload Multiple Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch")
    
    if uploaded_files:
        batch_results = []
        progress_bar = st.progress(0)
        
        # 1. Process each image
        for i, file in enumerate(uploaded_files):
            img = Image.open(file)
            results = model.predict(source=img, conf=conf_threshold)
            count = len(results[0].boxes)
            
            # Create a dictionary for each row
            batch_results.append({
                "No.": i + 1,
                "Photo Name": file.name,
                "Wheat Detected": count
            })
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # 2. Convert to a Pandas DataFrame for display and export
        df = pd.DataFrame(batch_results)
        
        st.success(f"✅ Processed {len(uploaded_files)} images successfully!")
        
        # 3. Show the table in the app
        st.dataframe(df, use_container_width=True)

        # 4. Create the CSV Download Button
        st.divider()
        csv = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Results as CSV (Excel Compatible)",
            data=csv,
            file_name="wheat_detection_report.csv",
            mime="text/csv",
            help="Click to download the counts for all uploaded images."
        )
# --- TAB 3: CONTACT & PROFESSIONAL INFO ---
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Using columns to center the contact card
    _, center_col, _ = st.columns([0.1, 2.8, 0.1])
    
    with center_col:
        # st.markdown("<h2 style='text-align: center; color: #0033A0;'>Get in Touch</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666;'>Have questions about WheatVision AI or my research? Send a message below.</p>", unsafe_allow_html=True)
        
        with st.container(border=True):
            # Header with your name
            st.markdown("<h4 style='background-color: #0033A0; text-align: center; color: white; padding: 10px; border-radius: 5px;'>Mohammad Shamim</h4>", unsafe_allow_html=True)
            
            # Professional Bio
            st.markdown("""
                <div style='text-align: center; margin-bottom: 20px;'>
                    <p style='margin: 10px 0;'><b>Crop Physiologist / Agronomist / Grain Crops Extension Associate</b></p>
                    <p style='font-size: 0.9em; color: #555;'>Specializing in eco-physiological responses, Data Analytics, & Decision Support Tools</p>
                    <p style='font-size: 0.9em; color: #555; font-weight: bold;'>University of Kentucky</p>
                </div>
            """, unsafe_allow_html=True)
            
            # The FormSubmit.co Contact Form
            contact_form = """
            <form action="https://formsubmit.co/shamim.one@outlook.com" method="POST">
                <input type="hidden" name="_captcha" value="false">
                <div style="margin-bottom: 10px;">
                    <input type="text" name="name" placeholder="Your Name" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ccc;" required>
                </div>
                <div style="margin-bottom: 10px;">
                    <input type="email" name="email" placeholder="Email Address" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ccc;" required>
                </div>
                <div style="margin-bottom: 10px;">
                    <textarea name="message" placeholder="How can I help you with your wheat analysis?" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ccc; height: 100px;"></textarea>
                </div>
                <button type="submit" style="background-color: #0033A0; color: white; border: none; padding: 12px 20px; border-radius: 5px; cursor: pointer; width: 100%; font-weight: bold;">
                    🚀 Send Message
                </button>
            </form>
            """
            st.markdown(contact_form, unsafe_allow_html=True)

        # Footer Links
        st.markdown("""
            <div style='text-align: center; margin-top: 20px;'>
                <a href="mailto:shamim.one@outlook.com" style="text-decoration: none; color: #0033A0; font-weight: bold;">📧 Email</a> | 
                <a href="https://github.com/shamim-mj" style="text-decoration: none; color: #0033A0; font-weight: bold;">💻 GitHub</a>
            </div>
        """, unsafe_allow_html=True)
