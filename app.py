import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io
import os
from src.models.pair_model import PAIRModel, TextExtractor, detect_adversarial_image, extract_hidden_text
from src.utils.image_processing import (
    load_image,
    apply_perturbation,
    detect_edges,
    extract_text_regions,
    visualize_results
)

# Set page config
st.set_page_config(
    page_title="Adversarial Image Detection",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state for models
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PAIRModel().to(device)
    text_extractor = TextExtractor().to(device)
    
    # Load saved models if they exist
    if os.path.exists('models/pair_model.pth'):
        model.load_state_dict(torch.load('models/pair_model.pth', map_location=device))
    if os.path.exists('models/text_extractor.pth'):
        text_extractor.load_state_dict(torch.load('models/text_extractor.pth', map_location=device))
    
    return model, text_extractor

def process_uploaded_image(image, model, text_extractor):
    """Process uploaded image and return results"""
    # Convert uploaded file to image
    image = Image.open(image)
    image = np.array(image)
    
    # Save temporarily
    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Process image
    image_tensor = load_image(temp_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = image_tensor.to(device)
    
    # Detect if image is adversarial
    is_adversarial, confidence = detect_adversarial_image(model, image_tensor)
    
    # Extract text regions
    text_regions = extract_text_regions(image_tensor)
    
    # Extract hidden text if image is adversarial
    hidden_text = None
    if is_adversarial:
        text_features = extract_hidden_text(text_extractor, image_tensor)
        hidden_text = text_features
    
    # Visualize results
    visualization = visualize_results(image_tensor, is_adversarial, confidence, text_regions)
    
    # Clean up
    os.remove(temp_path)
    
    return {
        'is_adversarial': is_adversarial,
        'confidence': confidence,
        'text_regions': text_regions,
        'hidden_text': hidden_text,
        'visualization': visualization
    }

def main():
    st.title("🛡️ Adversarial Image Detection")
    st.markdown("""
    This application uses the PAIR (Perturbation Analysis and Image Recognition) method to detect adversarial images
    that could be used for jailbreaking AI models. It can also identify and extract hidden text within images.
    """)
    
    # Load models
    try:
        model, text_extractor = load_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please ensure the model files are present in the models directory.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # Process image
        try:
            results = process_uploaded_image(uploaded_file, model, text_extractor)
            
            with col2:
                st.subheader("Analysis Results")
                st.image(results['visualization'], use_column_width=True)
                
                # Display results
                st.markdown("### Detection Results")
                status_color = "🔴" if results['is_adversarial'] else "🟢"
                st.markdown(f"**Status:** {status_color} {'Adversarial' if results['is_adversarial'] else 'Safe'}")
                st.markdown(f"**Confidence:** {results['confidence']:.2%}")
                
                if results['text_regions']:
                    st.markdown(f"**Text Regions Detected:** {len(results['text_regions'])}")
                
                if results['hidden_text'] is not None:
                    st.markdown("### Hidden Text Analysis")
                    st.markdown("Text features detected in the image:")
                    st.code(results['hidden_text'])
                
                # Additional analysis
                st.markdown("### Additional Analysis")
                if results['is_adversarial']:
                    st.warning("⚠️ This image shows characteristics of adversarial manipulation.")
                    st.info("Consider reviewing the image carefully and verifying its source.")
                else:
                    st.success("✅ This image appears to be safe and free from adversarial manipulation.")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Add information about the model
    with st.expander("About the Model"):
        st.markdown("""
        ### PAIR Method
        The PAIR (Perturbation Analysis and Image Recognition) method combines:
        - Deep learning-based image classification
        - Perturbation analysis for detecting adversarial patterns
        - Text extraction and analysis
        - Robust feature extraction
        
        ### How it Works
        1. The model analyzes the image for unusual patterns and perturbations
        2. It detects potential text regions and hidden messages
        3. The system combines multiple features to make a robust decision
        4. Results are visualized with confidence scores and detected regions
        
        ### Use Cases
        - Detecting adversarial images in AI systems
        - Identifying hidden text in images
        - Analyzing image authenticity
        - Preventing AI model jailbreaking attempts
        """)

if __name__ == "__main__":
    main() 