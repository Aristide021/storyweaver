"""
Stable Diffusion XL model inference for image generation.
"""

from typing import List, Optional, Dict, Any
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from .utils import get_device, setup_logging

logger = setup_logging()


class SDXLInferencer:
    """Stable Diffusion XL model for generating images from text prompts."""
    
    def __init__(self, model_name: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        """Initialize SDXL pipeline."""
        self.device = get_device()
        logger.info(f"Loading SDXL model {model_name} on {self.device}")
        
        # Load the pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if self.device.type == "cuda" else None
        )
        
        # Use DPM Solver for faster inference
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        self.pipeline = self.pipeline.to(self.device)
        
        # Enable memory efficient attention if available
        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()
        
        if hasattr(self.pipeline, "enable_model_cpu_offload") and self.device.type == "cuda":
            self.pipeline.enable_model_cpu_offload()
    
    def create_image_prompt(self, story: str, attributes: Dict[str, Any] = None) -> str:
        """Create an image generation prompt from story text and attributes."""
        # Extract key visual elements from the story
        prompt_parts = []
        
        # Start with a base style
        prompt_parts.append("high quality, detailed, cinematic lighting")
        
        # Add story-based elements
        if story:
            # Simple keyword extraction (in practice, you might want more sophisticated NLP)
            story_lower = story.lower()
            
            # Extract potential subjects
            subjects = []
            if any(word in story_lower for word in ["character", "person", "man", "woman", "child"]):
                subjects.append("character")
            if any(word in story_lower for word in ["forest", "trees", "woods"]):
                subjects.append("forest scene")
            if any(word in story_lower for word in ["castle", "palace", "tower"]):
                subjects.append("castle")
            if any(word in story_lower for word in ["ocean", "sea", "water", "beach"]):
                subjects.append("ocean scene")
            
            if subjects:
                prompt_parts.extend(subjects)
        
        # Add attributes if provided
        if attributes:
            caption = attributes.get('caption', '')
            if caption:
                prompt_parts.append(caption)
            
            # Add other relevant attributes
            for key, value in attributes.items():
                if key in ['What colors are prominent in this image?', 'What is the mood or atmosphere?']:
                    prompt_parts.append(value)
        
        # Add quality enhancers
        prompt_parts.extend([
            "masterpiece", "best quality", "8k resolution",
            "professional photography", "sharp focus"
        ])
        
        return ", ".join(prompt_parts)
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, ugly, bad anatomy",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Generate a single image from a text prompt."""
        
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        logger.info(f"Generating image with prompt: {prompt[:100]}...")
        
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
        
        image = result.images[0]
        logger.info(f"Generated image of size {image.size}")
        return image
    
    def generate_story_images(
        self,
        story: str,
        attributes: Dict[str, Any] = None,
        num_images: int = 3,
        **generation_kwargs
    ) -> List[Image.Image]:
        """Generate multiple images for a story."""
        base_prompt = self.create_image_prompt(story, attributes)
        
        images = []
        for i in range(num_images):
            logger.info(f"Generating image {i+1}/{num_images}")
            
            # Add variation to each image
            if i == 0:
                prompt = base_prompt
            elif i == 1:
                prompt = f"{base_prompt}, close-up view, dramatic angle"
            else:
                prompt = f"{base_prompt}, wide shot, landscape view"
            
            image = self.generate_image(prompt, **generation_kwargs)
            images.append(image)
        
        return images
    
    def generate_variations(
        self,
        base_prompt: str,
        num_variations: int = 4,
        **generation_kwargs
    ) -> List[Image.Image]:
        """Generate variations of the same prompt with different seeds."""
        images = []
        
        for i in range(num_variations):
            logger.info(f"Generating variation {i+1}/{num_variations}")
            
            # Use different seeds for variation
            generation_kwargs['seed'] = i * 1000 if 'seed' not in generation_kwargs else generation_kwargs['seed'] + i
            
            image = self.generate_image(base_prompt, **generation_kwargs)
            images.append(image)
        
        return images
    
    def upscale_image(self, image: Image.Image, scale_factor: float = 2.0) -> Image.Image:
        """Simple upscaling using PIL (for basic upscaling)."""
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        upscaled = image.resize(new_size, Image.LANCZOS)
        logger.info(f"Upscaled image from {image.size} to {upscaled.size}")
        return upscaled 