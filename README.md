# 🌾 WheatVision AI: Precision Wheat Head Detection & Yield Assessment

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.app)
[![University of Kentucky](https://img.shields.io/badge/University%20of-Kentucky-003DA5?logo=data:image/png;base64,)](https://uky.edu)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/Model-YOLOv11-green)](https://ultralytics.com)

**WheatVision AI** is a professional-grade decision support tool designed to automate the detection and counting of wheat heads from field imagery. Developed for agronomists, researchers, and producers, this tool leverages state-of-the-art Computer Vision (YOLOv11) to provide rapid, accurate yield assessments directly from smartphone or drone photos.

---

## 🚀 Key Features

- **High-Precision Detection** — Trained on a massive dataset featuring hundreds of thousands of wheat head instances (mAP50: 96%).
- **Dual Analysis Modes:**
  - 🎯 **Single Analysis** — Upload or capture a field photo for immediate visual feedback and head count.
  - 📂 **Batch Processing** — Process multiple images simultaneously and export a full `.csv` summary report.
- **📐 Automatic Area & Density Calculation:**
  - Supports **38 phone models** (iPhone 12–18 Pro Max, Samsung S21–S26 Ultra, A-series, Google Pixel, and more).
  - **Auto-detects your phone** from photo EXIF metadata — no manual input required.
  - Falls back to a manual phone selector if metadata is unavailable.
  - Calculates **ground area (m² and ft²)**, **GSD (cm/pixel)**, and **heads per m²/ha/acre**.
- **📈 Yield Estimator** — Built-in agronomy calculator to predict **Bushels per Acre** based on detected head density, thousand-grain weight, and sample area (auto-filled from area calculation).
- **📱 Mobile Optimized** — Clear field photo guide, native camera workflow, and responsive layout for smartphone use.
- **🚁 Drone Compatible** — Reads focal length directly from drone EXIF (e.g., DJI M3M, Mavic 3) for accurate area calculation.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Accuracy (mAP50) | **96%** |
| Overall Recall | **92%** |
| Max Detections per Image | 2,000 heads |
| Optimized For | Overlapping heads, variable lighting, dense canopies |

---

## 🛠️ How to Use

### 📱 Field Photo Protocol (for accurate area calculation)
1. Open your phone's **native Camera app**
2. Hold phone **straight above** wheat heads at a **fixed height** (e.g., 0.3m ≈ 1 ft)
3. Point camera **straight down (90°)** — use phone level app if needed
4. Take the photo, then upload it in the app
5. Set the **height slider** in the sidebar to match how high you held the phone

### 🖥️ App Workflow
1. **Set Height** — Adjust the sidebar slider to your phone height above the crop
2. **Upload** — Drop your field photo into the **Single Analysis** or **Batch Processing** tab
3. **Auto-detect** — App reads your phone model from EXIF and calculates ground area automatically
4. **Analyze** — Review head count, area, density (heads/m²), and annotated imagery
5. **Estimate Yield** — Use the built-in yield predictor (optional)
6. **Export** — Download a full CSV report with area, density, and GSD per image

### ⚙️ Sensitivity Slider
- Recommended range: `0.15 – 0.35`
- **Lower values** → detect more heads (may include false positives)
- **Higher values** → stricter detection (may miss some heads)

---

## 📐 Area Calculation — How It Works

WheatVision uses the **Ground Sampling Distance (GSD)** formula:

```
Area = (sensor_width × sensor_height / focal_length²) × height²
```

This is **resolution-independent** — only the phone's optics and your height above the crop determine the area. The app auto-reads focal length from EXIF when available, or uses a curated database of 38 phone models as fallback.

### Supported Camera Sources
| Source | How It Works |
|---|---|
| iPhone 12–18 Pro Max | Auto-detected from EXIF |
| Samsung S21–S26 Ultra | Auto-detected via Samsung model codes (e.g., SM-A356 = A35) |
| Samsung A25, A35, A54, A55 | Auto-detected |
| Google Pixel 7–9 | Auto-detected |
| DJI drones (M3M, Mavic 3, etc.) | Focal length read directly from EXIF |
| Any other camera | Manual selection from dropdown |

---

## 📦 Installation (Local Use)

### 1. Clone the repository
```bash
git clone https://github.com/shamim-mj/wheatvision.git
cd wheatvision
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

> **Note:** Ensure `best.pt` (the YOLO model file) is in the same folder as `app.py`.

---

## 📋 Requirements

```txt
streamlit>=1.40.0
opencv-python-headless>=4.9.0.80
ultralytics>=8.3.0
pillow>=10.2.0
pandas>=2.2.2
numpy>=1.26.4
torch>=2.2.2
torchvision>=0.17.0
rich>=13.9.4
```

> 💡 **Streamlit Cloud users:** If you hit memory limits, use CPU-only PyTorch:
> ```
> --extra-index-url https://download.pytorch.org/whl/cpu
> torch==2.2.2+cpu
> torchvision==0.17.2+cpu
> ```

---

## 📁 Project Structure

```
wheatvision/
├── app.py                  # Main Streamlit application
├── best.pt                 # YOLOv11 trained model weights
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 👨‍🔬 About the Developer

**Mohammad Jan Shamim, Ph.D.**
Crop Physiologist / Agronomist / Grain Crops Extension & Research Associate
University of Kentucky

Specializing in eco-physiological responses of crop species, precision agriculture, data analytics, and the development of decision support tools for modern agriculture.

📧 [shamim.one@outlook.com](mailto:shamim.one@outlook.com)
💻 [github.com/shamim-mj](https://github.com/shamim-mj)

---

## ⚠️ Disclaimer

This tool is intended for **research and extension purposes**. Environmental factors, crop variety, canopy structure, and image quality may impact individual results. Always validate outputs with ground-truth counts when making critical management decisions.

---

*Built with ❤️ for wheat growers and agricultural researchers.*
