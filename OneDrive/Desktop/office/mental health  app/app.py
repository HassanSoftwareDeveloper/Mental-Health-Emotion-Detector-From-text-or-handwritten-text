import streamlit as st
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
from ocr_utils import extract_text_from_image
from nlp_clean import clean_text
from emotion_model import detect_emotions




# Initialize the app
st.set_page_config(
    page_title="Mental Health Emotion Analyzer", 
    layout="centered",
    page_icon="🧠"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stTextArea textarea {
        font-family: monospace;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("🧠 Mental Health Emotion Analyzer")
st.markdown("""
This app analyzes handwritten or printed text to detect emotional states and mental health indicators.

**How it works:**
1. Upload an image containing text
2. The app extracts text using advanced OCR
3. NLP techniques clean and process the text
4. Emotion analysis detects psychological states
5. Results are visualized for easy interpretation
""")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    ocr_mode = st.selectbox(
        "OCR Mode", 
        ["Standard", "Enhanced", "Gemini (Best for handwriting)"], 
        help="Standard: Good for printed text\nEnhanced: Better for difficult images\nGemini: Best for handwriting"
    ).lower()
    
    show_steps = st.checkbox("Show image processing steps", value=False)
    debug_mode = st.checkbox("Enable debug mode", value=False)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool is for educational purposes only and not a substitute for professional mental health care.
    """)
    st.markdown("---")
    st.caption("Mental Health Analyzer v2.0.0")

def plot_emotions(emotions):
    """Enhanced emotion visualization with better formatting"""
    if not emotions:
        st.warning("No emotions detected")
        return
    
    # Sort emotions by score
    emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    labels = [emo['label'] for emo in emotions]
    scores = [emo['score'] for emo in emotions]
    
    # Color mapping based on emotion type
    color_map = {
        'positive': '#4CAF50',
        'negative': '#F44336',
        'neutral': '#2196F3'
    }
    
    colors = []
    for label in labels:
        label_lower = label.lower()
        if label_lower in ['happiness', 'joy', 'love', 'excitement']:
            colors.append(color_map['positive'])
        elif label_lower in ['anger', 'fear', 'sadness', 'anxiety', 'depression']:
            colors.append(color_map['negative'])
        else:
            colors.append(color_map['neutral'])
    
    bars = ax.barh(labels, scores, color=colors)
    ax.set_xlim(0, 1)
    ax.set_title('Emotional State Analysis', pad=20, fontsize=14)
    ax.set_xlabel('Confidence Score', labelpad=10, fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value annotations
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}',
                ha='left', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)

def show_mental_health_indicators(emotions):
    """Display mental health indicator cards"""
    if not emotions:
        return
    
    st.markdown("### 🩺 Mental Health Indicators")
    
    # Define important indicators
    indicators = {
        'Depression': ['Sadness', 'Hopelessness'],
        'Anxiety': ['Anxiety', 'Fear'],
        'Stress': ['Anger', 'Frustration'],
        'Well-being': ['Happiness', 'Contentment']
    }
    
    # Calculate scores for each indicator
    indicator_scores = {}
    for name, tags in indicators.items():
        relevant_emotions = [e for e in emotions if e['label'] in tags]
        if relevant_emotions:
            score = max(e['score'] for e in relevant_emotions)
            indicator_scores[name] = score
    
    # Display in columns
    cols = st.columns(len(indicator_scores))
    for i, (name, score) in enumerate(indicator_scores.items()):
        with cols[i]:
            if name == 'Well-being':
                color = "green" if score > 0.5 else "orange"
                delta = "High" if score > 0.5 else "Moderate"
            else:
                color = "red" if score > 0.5 else "orange" if score > 0.3 else "green"
                delta = "High concern" if score > 0.5 else "Moderate" if score > 0.3 else "Low"
            
            st.metric(
                label=name,
                value=f"{score*100:.0f}%",
                delta=delta,
                delta_color="inverse" if name != 'Well-being' else "normal"
            )

# Main app logic
uploaded_file = st.file_uploader(
    "📤 Upload an image (JPEG, PNG)", 
    type=["jpg", "jpeg", "png"],
    help="Upload an image containing handwritten or printed text"
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Display uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Processing..."):
                start_time = time.time()
                
                # Debug output
                if debug_mode:
                    st.write(f"Running OCR in {ocr_mode} mode...")
                
                # Extract text
                extracted_text = extract_text_from_image(
                    image_np, 
                    mode=ocr_mode, 
                    show_steps=show_steps,
                    debug=debug_mode
                )
                
                # Show extracted text
                st.subheader("📝 Extracted Text")
                st.text_area("Raw OCR Output", extracted_text, height=150, label_visibility="collapsed")
                
                # Process text through pipeline
                cleaned_text, primary_mood, emotion_scores = process_text(extracted_text)
                
                st.subheader("🧹 Cleaned Text")
                st.text_area("After NLP Processing", cleaned_text, height=100, label_visibility="collapsed")
                
                # Emotion detection
                st.subheader("😊 Emotion Analysis")
                if cleaned_text.strip():
                    try:
                        # Get emotions from emotion_model
                        emotions = detect_emotions(cleaned_text)
                        
                        # Plot emotions
                        plot_emotions(emotions)
                        
                        # Display primary mood
                        st.markdown(f"**Primary Mood Detected:** {primary_mood}")
                        
                        # Generate and display summary
                        summary = generate_emotion_summary(emotions)
                        st.info(summary)
                        
                        # Mental health indicators
                        show_mental_health_indicators(emotions)
                        
                    except Exception as e:
                        st.error(f"Emotion detection failed: {str(e)}")
                        if debug_mode:
                            st.exception(e)
                else:
                    st.warning("No meaningful text found for emotion analysis")
                
                st.success(f"✅ Analysis completed in {time.time()-start_time:.2f} seconds")
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        if debug_mode:
            st.exception(e)