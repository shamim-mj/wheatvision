# 🌾 WheatVision AI: Precision Wheat Head Detection & Yield Assessment

[![Streamlit App](https://streamlit.io)](https://streamlit.app)
[![University of Kentucky](https://shields.io)](https://uky.edu)

**WheatVision AI** is a professional-grade decision support tool designed to automate the detection and counting of wheat heads from field imagery. Developed for agronomists, researchers, and producers, this tool leverages state-of-the-art Computer Vision (YOLOv11) to provide rapid yield assessments.

---

## 🚀 Key Features

- **High-Precision Detection:** Trained on a massive dataset featuring hundreds of thousands of wheat head instances.
- **Dual Analysis Modes:**
  - **Single Analysis:** Upload or capture a field photo for immediate visual feedback.
  - **Batch Processing:** Process multiple images simultaneously and export a `.csv` summary.
- **Yield Estimator:** Built-in agronomy calculator to predict **Bushels per Acre** based on detected density, grain weight, and sample area.
- **Mobile Optimized:** Use your smartphone camera directly in the field for real-time data collection.

## 📊 Model Performance

Our model has been rigorously validated to ensure reliable field performance:
- **Accuracy (mAP50):** 96% (at 50% Intersection over Union).
- **Overall Recall:** 92% (successfully identifies 92 out of every 100 wheat heads).
- **Optimized for Density:** Specifically tuned to handle overlapping wheat heads and variable lighting conditions.

---

## 🛠️ How to Use

1. **Upload:** Drop your field photos into the **Single Analysis** or **Batch Processing** tabs.
2. **Adjust:** Use the **Sensitivity Slider** in the sidebar to fine-tune detections (Recommended: `0.25 - 0.35`).
3. **Analyze:** Review the automated count and annotated imagery.
4. **Export:** Download your annotated images or a full CSV report for your field records.

---

## 📦 Installation (Local Use)

If you wish to run this tool locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shamim-mj/wheatvision.git
   cd wheatvision

2. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
4. **Run the app:**
```bash
  streamlit run app.py
   <br><br><br>
**👨‍🔬 About the Developer:**
Mohammad Shamim
Crop Physiologist / Agronomist / Grain Crops Extension & Research Associate
University of Kentucky
Specializing in eco-physiological responses of crop species, Data Analytics, and the development of Decision Support Tools for modern agriculture.
**Note:** This tool is intended for research and extension purposes. Environmental factors and crop variety may impact individual results.
