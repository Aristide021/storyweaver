"""
Smoke tests for the Storyweaver pipeline components.
These tests verify basic functionality without requiring GPU resources.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import get_device, load_image, validate_inputs, setup_logging


class TestUtils:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device.type in ['cuda', 'mps', 'cpu']
    
    def test_validate_inputs(self):
        """Test input validation."""
        inputs = {'key1': 'value1', 'key2': 'value2'}
        required_keys = ['key1', 'key2']
        
        # Should not raise an exception
        validate_inputs(inputs, required_keys)
        
        # Should raise an exception for missing keys
        with pytest.raises(ValueError):
            validate_inputs(inputs, ['key1', 'key2', 'missing_key'])
    
    def test_setup_logging(self):
        """Test logging setup."""
        logger = setup_logging()
        assert logger is not None
        assert logger.name == 'utils'


class TestBLIPInferencer:
    """Test BLIP inference functionality."""
    
    @patch('src.infer_blip.BlipProcessor')
    @patch('src.infer_blip.BlipForConditionalGeneration')
    def test_blip_initialization(self, mock_model, mock_processor):
        """Test BLIP model initialization."""
        from infer_blip import BLIPInferencer
        
        # Mock the model and processor
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        
        blip = BLIPInferencer()
        
        assert blip.processor is not None
        assert blip.model is not None
        assert blip.device is not None
    
    def test_create_test_image(self):
        """Create a test image for other tests."""
        # Create a simple test image
        test_image = Image.new('RGB', (256, 256), color='red')
        return test_image


class TestLlamaInferencer:
    """Test LLaMA inference functionality."""
    
    @patch('src.infer_llama.AutoTokenizer')
    @patch('src.infer_llama.AutoModelForCausalLM')
    def test_llama_initialization(self, mock_model, mock_tokenizer):
        """Test LLaMA model initialization."""
        from infer_llama import LlamaInferencer
        
        # Mock the tokenizer and model
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        
        # Mock tokenizer attributes
        mock_tokenizer_instance = mock_tokenizer.from_pretrained.return_value
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = '<eos>'
        
        llama = LlamaInferencer()
        
        assert llama.tokenizer is not None
        assert llama.model is not None
        assert llama.device is not None
    
    def test_create_story_prompt(self):
        """Test story prompt creation."""
        from infer_llama import LlamaInferencer
        
        # Create a mock LlamaInferencer without initializing models
        with patch('src.infer_llama.AutoTokenizer'), \
             patch('src.infer_llama.AutoModelForCausalLM'):
            
            llama = LlamaInferencer()
            
            # Test prompt creation
            attributes = {
                'caption': 'A beautiful landscape',
                'What is the main subject in this image?': 'mountains',
                'What colors are prominent in this image?': 'green and blue'
            }
            
            prompt = llama.create_story_prompt(attributes)
            
            assert 'beautiful landscape' in prompt
            assert 'mountains' in prompt
            assert 'Story:' in prompt


class TestSDXLInferencer:
    """Test SDXL inference functionality."""
    
    @patch('src.infer_sdxl.StableDiffusionXLPipeline')
    def test_sdxl_initialization(self, mock_pipeline):
        """Test SDXL model initialization."""
        from infer_sdxl import SDXLInferencer
        
        # Mock the pipeline
        mock_pipeline.from_pretrained.return_value = MagicMock()
        
        sdxl = SDXLInferencer()
        
        assert sdxl.pipeline is not None
        assert sdxl.device is not None
    
    def test_create_image_prompt(self):
        """Test image prompt creation."""
        from infer_sdxl import SDXLInferencer
        
        # Create a mock SDXLInferencer without initializing models
        with patch('src.infer_sdxl.StableDiffusionXLPipeline'):
            sdxl = SDXLInferencer()
            
            story = "In a magical forest, ancient trees whispered secrets to the wind."
            attributes = {
                'caption': 'A mystical forest scene',
                'What colors are prominent in this image?': 'green and brown'
            }
            
            prompt = sdxl.create_image_prompt(story, attributes)
            
            assert 'forest' in prompt.lower()
            assert 'green' in prompt.lower()
            assert 'high quality' in prompt.lower()


class TestPipeline:
    """Test the main pipeline integration."""
    
    @patch('src.pipeline.BLIPInferencer')
    @patch('src.pipeline.LlamaInferencer')
    @patch('src.pipeline.SDXLInferencer')
    @patch('src.pipeline.mlflow')
    def test_pipeline_initialization(self, mock_mlflow, mock_sdxl, mock_llama, mock_blip):
        """Test pipeline initialization."""
        from pipeline import StoryweaverPipeline
        
        # Mock the inferencers
        mock_blip.return_value = MagicMock()
        mock_llama.return_value = MagicMock()
        mock_sdxl.return_value = MagicMock()
        
        pipeline = StoryweaverPipeline()
        
        assert pipeline.blip is not None
        assert pipeline.llama is not None
        assert pipeline.sdxl is not None


def test_project_structure():
    """Test that all required files and directories exist."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # Check main files
    assert os.path.exists(os.path.join(project_root, 'README.md'))
    assert os.path.exists(os.path.join(project_root, 'requirements.txt'))
    
    # Check src directory
    src_dir = os.path.join(project_root, 'src')
    assert os.path.exists(src_dir)
    assert os.path.exists(os.path.join(src_dir, 'utils.py'))
    assert os.path.exists(os.path.join(src_dir, 'pipeline.py'))
    assert os.path.exists(os.path.join(src_dir, 'infer_blip.py'))
    assert os.path.exists(os.path.join(src_dir, 'infer_llama.py'))
    assert os.path.exists(os.path.join(src_dir, 'infer_sdxl.py'))
    
    # Check notebooks directory
    notebooks_dir = os.path.join(project_root, 'notebooks')
    assert os.path.exists(notebooks_dir)
    assert os.path.exists(os.path.join(notebooks_dir, '01_extract_attributes.ipynb'))
    assert os.path.exists(os.path.join(notebooks_dir, '02_generate_copy.ipynb'))
    assert os.path.exists(os.path.join(notebooks_dir, '03_generate_images.ipynb'))
    
    # Check demo directory
    demo_dir = os.path.join(project_root, 'demo')
    assert os.path.exists(demo_dir)
    assert os.path.exists(os.path.join(demo_dir, 'app.py'))
    
    # Check tests directory
    tests_dir = os.path.join(project_root, 'tests')
    assert os.path.exists(tests_dir)
    assert os.path.exists(os.path.join(tests_dir, 'smoke_test.py'))


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])