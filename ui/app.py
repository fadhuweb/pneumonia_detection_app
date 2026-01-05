import streamlit as st
import os
import tempfile
from PIL import Image
import sys

# Add the root directory to sys.path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_trained_model
from src.prediction import predict_image_from_path

st.set_page_config(
    page_title="Pneumonia Detection",
    layout="centered",
    page_icon="🏥"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }
    .pneumonia {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
    }
    .normal {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-align: center;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 30px;
        border-radius: 15px;
        border: 2px dashed #dee2e6;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    """Find and load the latest model file once."""
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    if not os.path.exists(MODELS_DIR):
        return None
        
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(('.keras', '.h5'))]
    if not model_files:
        return None
    
    # Sort by modification time to get the latest
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)), reverse=True)
    latest_model = os.path.join(MODELS_DIR, model_files[0])
    return load_trained_model(latest_model)

# Header
st.title("🏥 Chest X-ray Pneumonia Detection")
st.markdown("---")
st.markdown("Upload a chest X-ray image to detect pneumonia using AI")

# Pre-load model
model = get_model()
if model is None:
    st.error("❌ Model file not found in the `models/` directory.")
    st.info("Please ensure your `.keras` model file is in the `models/` folder.")

# File uploader
st.markdown("### 📤 Upload Image")
file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if file:
    # Display uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Uploaded Image")
        image = Image.open(file)
        st.image(image, use_container_width=True)
        st.caption(f"📁 {file.name} ({file.size / 1024:.1f} KB)")
    
    with col2:
        st.markdown("#### Prediction")
        
        if st.button("🔍 Analyze Image", use_container_width=True):
            if model is None:
                st.error("Prediction cannot run because the model is missing.")
            else:
                with st.spinner("Analyzing X-ray..."):
                    try:
                        # Save uploaded image to temp file (optional but safe)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                            tmp.write(file.getvalue())
                            tmp_path = tmp.name
                        
                        # Run prediction directly using the brain logic
                        result = predict_image_from_path(model, tmp_path)
                        os.unlink(tmp_path)
                        
                        prediction = result.get('label', 'Unknown')
                        
                        if prediction.upper() == "PNEUMONIA":
                            st.markdown(f"""
                                <div class="result-box pneumonia">
                                    <h2>⚠️ PNEUMONIA DETECTED</h2>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class="result-box normal">
                                    <h2>✅ NORMAL</h2>
                                </div>
                            """, unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
        else:
            st.info("👆 Click the button above to analyze the image")

else:
    st.info("👆 Upload a chest X-ray image to get started")
