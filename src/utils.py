"""
Utility functions for the storyweaver pipeline.
"""

import os
import json
from typing import Dict, Any, List
from PIL import Image
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save pipeline results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_image(image_path: str) -> Image.Image:
    """Load and validate image file."""
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def validate_inputs(inputs: Dict[str, Any], required_keys: List[str]) -> None:
    """Validate that all required keys are present in inputs."""
    missing_keys = [key for key in required_keys if key not in inputs]
    if missing_keys:
        raise ValueError(f"Missing required inputs: {missing_keys}")


def setup_logging():
    """Setup logging configuration."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__) 