
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from ocr_utils import extract_text_from_image
from nlp_clean import clean_text
from emotion_model import detect_emotions

# Initialize the app
st.set_page_config(page_title="Mental Health Emotion Analyzer", layout="centered")
st.title("Mental Health Emotion Analyzer")
st.markdown("""
Upload handwritten or printed images. This app will:
- Extract text using advanced OCR
- Clean the text using NLP
- Detect emotional indicators (Stress, Anxiety, Sadness, etc.)
- Visualize emotional state
""")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    ocr_mode = st.selectbox(
        "OCR Mode", 
        ["Standard", "Enhanced"], 
        help="Use 'Enhanced' for handwritten text"
    ).lower()
    show_steps = st.checkbox("Show processing steps", value=True)

def plot_emotions(emotions):
    """Visualize emotion analysis results"""
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = [emo['label'] for emo in emotions]
    scores = [emo['score'] for emo in emotions]
    
    colors = []
    for label in labels:
        if label.lower() in ['sadness', 'anger', 'fear']:
            colors.append('red')
        elif label.lower() in ['joy', 'love']:
            colors.append('green')
        else:
            colors.append('blue')
    
    bars = ax.barh(labels, scores, color=colors)
    ax.set_xlim(0, 1)
    ax.set_title('Emotional State Analysis')
    ax.set_xlabel('Confidence Score')
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}',
                ha='left', va='center')
    
    st.pyplot(fig)

# Main app logic
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Extract Text and Analyze"):
        with st.spinner("Processing..."):
            start_time = time.time()
            
            # Debug output
            st.write("Running OCR in", ocr_mode, "mode...")
            
            # Extract text
            extracted_text = extract_text_from_image(
                image_np, 
                mode=ocr_mode.lower(), 
                show_steps=show_steps
            )
            st.subheader("Extracted Text")
            st.text_area("Raw OCR Output", extracted_text, height=150)

            # Clean text
            cleaned_text = clean_text(extracted_text)
            st.subheader("Cleaned Text")
            st.text_area("After NLP Processing", cleaned_text, height=100)

            # Emotion detection
            st.subheader("Emotion Analysis")
            if cleaned_text.strip():
                try:
                    emotions = detect_emotions(cleaned_text)
                    plot_emotions(emotions)
                    
                    # Mental health indicators
                    mental_health_indicators = {
                        'sadness': next((e for e in emotions if e['label'].lower() == 'sadness'), None),
                        'anxiety': next((e for e in emotions if e['label'].lower() == 'fear'), None),
                        'stress': next((e for e in emotions if e['label'].lower() == 'anger'), None)
                    }
                    
                    st.markdown("### Mental Health Indicators")
                    cols = st.columns(3)
                    for i, (name, data) in enumerate(mental_health_indicators.items()):
                        if data:
                            score = data['score']
                            color = "red" if score > 0.5 else "orange" if score > 0.3 else "green"
                            cols[i].metric(
                                label=name.capitalize(),
                                value=f"{score*100:.1f}%",
                                delta="High concern" if score > 0.5 else "Moderate" if score > 0.3 else "Low",
                                delta_color="off"
                            )
                except Exception as e:
                    st.error(f"Emotion detection failed: {str(e)}")
            else:
                st.warning("No meaningful text found for emotion analysis")
            
            st.success(f"Analysis completed in {time.time()-start_time:.2f} seconds")

# Debug message to confirm correct version
st.sidebar.markdown("---")
st.sidebar.caption("App version 1.0.1")