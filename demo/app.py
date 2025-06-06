"""
Streamlit demo application for the Storyweaver pipeline.
"""

import streamlit as st
import sys
import os
from PIL import Image
import json
from io import BytesIO
import zipfile

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import StoryweaverPipeline
from utils import setup_logging

logger = setup_logging()

# Configure Streamlit page
st.set_page_config(
    page_title="Storyweaver",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 2rem;
}

.sub-header {
    font-size: 1.5rem;
    color: #2c5282;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

.story-box {
    background-color: #f7fafc;
    border-left: 4px solid #4299e1;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}

.attribute-box {
    background-color: #edf2f7;
    padding: 0.75rem;
    border-radius: 0.375rem;
    margin: 0.5rem 0;
}

.success-message {
    background-color: #f0fff4;
    border: 1px solid #9ae6b4;
    color: #22543d;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'results' not in st.session_state:
    st.session_state.results = None

def initialize_pipeline():
    """Initialize the pipeline with error handling."""
    try:
        with st.spinner("Initializing AI models... This may take a few minutes on first run."):
            st.session_state.pipeline = StoryweaverPipeline()
        return True
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return False

def display_attributes(attributes):
    """Display extracted attributes in a nice format."""
    st.markdown('<div class="sub-header">📋 Extracted Attributes</div>', unsafe_allow_html=True)
    
    for key, value in attributes.items():
        if key == 'caption':
            st.markdown(f"**Main Caption:** {value}")
        else:
            st.markdown(f"**{key}:** {value}")

def display_story(story):
    """Display the generated story."""
    st.markdown('<div class="sub-header">📖 Generated Story</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="story-box">{story}</div>', unsafe_allow_html=True)
    st.markdown(f"*Story length: {len(story.split())} words*")

def display_images(images):
    """Display generated images in a grid."""
    st.markdown('<div class="sub-header">🎨 Generated Images</div>', unsafe_allow_html=True)
    
    cols = st.columns(len(images))
    for i, img in enumerate(images):
        with cols[i]:
            st.image(img, caption=f"Generated Image {i+1}", use_column_width=True)

def create_download_package(results):
    """Create a ZIP file with all results for download."""
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add story
        story_content = results.get('story', '')
        zip_file.writestr('story.txt', story_content)
        
        # Add attributes
        attributes_content = json.dumps(results.get('attributes', {}), indent=2)
        zip_file.writestr('attributes.json', attributes_content)
        
        # Add images
        for i, img in enumerate(results.get('generated_images', [])):
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            zip_file.writestr(f'generated_image_{i+1}.png', img_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# Main app
def main():
    st.markdown('<div class="main-header">📚 Storyweaver</div>', unsafe_allow_html=True)
    st.markdown("Transform images into stories and bring stories to life with AI-generated visuals")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Model selection
        st.markdown("**Model Settings:**")
        use_default_models = st.checkbox("Use default models", value=True)
        
        if not use_default_models:
            blip_model = st.text_input("BLIP Model", "Salesforce/blip-image-captioning-base")
            llama_model = st.text_input("LLaMA Model", "meta-llama/Llama-2-7b-chat-hf")
            sdxl_model = st.text_input("SDXL Model", "stabilityai/stable-diffusion-xl-base-1.0")
        
        # Generation settings
        st.markdown("**Generation Settings:**")
        num_images = st.slider("Number of images to generate", 1, 5, 3)
        num_story_variants = st.slider("Number of story variants", 1, 3, 1)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            inference_steps = st.slider("SDXL Inference Steps", 10, 50, 20)
            guidance_scale = st.slider("SDXL Guidance Scale", 1.0, 20.0, 7.5)
            image_size = st.selectbox("Image Size", ["512x512", "768x768", "1024x1024"], index=2)
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to extract attributes and generate a story"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Initialize pipeline if needed
            if st.session_state.pipeline is None:
                if st.button("Initialize Pipeline", type="primary"):
                    initialize_pipeline()
            
            # Run pipeline button
            if st.session_state.pipeline is not None:
                if st.button("🚀 Generate Story & Images", type="primary"):
                    with st.spinner("Running the pipeline... This may take several minutes."):
                        try:
                            # Save uploaded image temporarily
                            temp_path = "temp_input.jpg"
                            image.save(temp_path)
                            
                            # Parse image size
                            width, height = map(int, image_size.split('x'))
                            
                            # Run pipeline
                            results = st.session_state.pipeline.run_full_pipeline(
                                temp_path,
                                num_story_variants=num_story_variants,
                                num_generated_images=num_images,
                                width=width,
                                height=height,
                                num_inference_steps=inference_steps,
                                guidance_scale=guidance_scale
                            )
                            
                            st.session_state.results = results
                            
                            # Clean up temp file
                            os.remove(temp_path)
                            
                            st.markdown('<div class="success-message">✅ Pipeline completed successfully!</div>', unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Pipeline failed: {e}")
                            logger.error(f"Pipeline error: {e}")
    
    with col2:
        if st.session_state.results:
            results = st.session_state.results
            
            # Display results
            if 'attributes' in results:
                display_attributes(results['attributes'])
            
            if 'story' in results:
                display_story(results['story'])
            
            if 'generated_images' in results:
                display_images(results['generated_images'])
            
            # Download button
            st.markdown("### 📥 Download Results")
            
            download_data = create_download_package(results)
            st.download_button(
                label="Download All Results (ZIP)",
                data=download_data,
                file_name="storyweaver_results.zip",
                mime="application/zip"
            )
        
        else:
            st.info("👆 Upload an image and run the pipeline to see results here")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    Built with ❤️ using Streamlit, BLIP, LLaMA, and Stable Diffusion XL
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 