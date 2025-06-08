"""
BLIP model inference for image attribute extraction.
"""

from typing import List, Dict, Any
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from .utils import get_device, setup_logging
from .download_and_load_models import ensure_all_models_downloaded, get_model_paths

logger = setup_logging()


class BLIPInferencer:
    """BLIP model for extracting attributes from images."""
    
    def __init__(self, model_name: str = None):
        """Initialize BLIP model and processor."""
        self.device = get_device()
        
        # Use cached model if no specific model name provided
        if model_name is None:
            logger.info("Using cached BLIP model...")
            ensure_all_models_downloaded()
            model_paths = get_model_paths()
            model_name = model_paths['blip']
            use_local = True
        else:
            use_local = False
            
        logger.info(f"Loading BLIP model {model_name} on {self.device}")
        
        self.processor = BlipProcessor.from_pretrained(model_name, local_files_only=use_local)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name, local_files_only=use_local)
        self.model.to(self.device)
        self.model.eval()
    
    def extract_caption(self, image: Image.Image) -> str:
        """Generate a caption for the image."""
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=50, num_beams=4)
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption
    
    def extract_attributes(self, image: Image.Image, questions: List[str] = None) -> Dict[str, Any]:
        """Extract attributes from image using visual question answering."""
        if questions is None:
            questions = [
                "What is the main subject in this image?",
                "What colors are prominent in this image?",
                "What is the setting or location?",
                "What is the mood or atmosphere?",
                "What objects are visible in this image?"
            ]
        
        attributes = {}
        
        # Generate general caption
        attributes['caption'] = self.extract_caption(image)
        
        # Answer specific questions
        for question in questions:
            inputs = self.processor(image, question, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=30, num_beams=4)
                answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the answer (remove the question if it's repeated)
            if question.lower() in answer.lower():
                answer = answer.replace(question, "").strip()
            
            attributes[question] = answer
        
        logger.info(f"Extracted {len(attributes)} attributes from image")
        return attributes
    
    def batch_extract_attributes(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Extract attributes from multiple images."""
        results = []
        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")
            attributes = self.extract_attributes(image)
            results.append(attributes)
        return results 