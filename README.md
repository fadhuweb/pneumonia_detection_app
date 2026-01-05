# Pneumonia Detection App

A deep learning application for detecting pneumonia from chest X-ray images. Built with Streamlit and TensorFlow (VGG16 Transfer Learning).

## Key Features
- **Clean UI:** Single-page interface focused on image upload and analysis.
- **Standalone Architecture:** Backend logic merged into the frontend for easier deployment.
- **Fast Inference:** Optimized for quick predictions without a separate API server.

## Local Setup

### Prerequisites
- Python 3.12 (Recommended for TensorFlow compatibility)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/fadhuweb/pneumonia_detection_app.git
   ```
2. Navigate to the folder:
   ```bash
   cd pneumonia_detection_app
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
```bash
streamlit run ui/app.py
```

## Deployment to Streamlit Cloud 🚀

This app is ready for one-click deployment:
1. Push this code to your GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account.
4. Select this repository and set the main file path to `ui/app.py`.
5. Click **Deploy!**

---

## Technical Details
- **Model:** VGG16 based Transfer Learning model.
- **Input:** 224x224 grayscale/RGB chest X-ray images.
- **Output:** Binary classification (PNEUMONIA or NORMAL).
