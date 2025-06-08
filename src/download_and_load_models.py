"""
Download and load models with intelligent caching for the Storyweaver pipeline.
This ensures models are only downloaded once per workspace/container.
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
import torch
from typing import Tuple, Any

# Import utils with fallback for standalone execution
try:
    from .utils import get_device, setup_logging
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils import get_device, setup_logging

logger = setup_logging()

# ──────────────────────────────────────────────────────────────────────────────
# 1) DOWNLOAD HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def safe_snapshot_download(repo_id: str, local_dir: str, exist_check_file: str):
    """
    Download a HF repo to local_dir if exist_check_file is missing.
    
    Args:
        repo_id: HuggingFace model repository ID
        local_dir: Local directory to store the model
        exist_check_file: Marker file that indicates successful download
    """
    local_path = Path(local_dir)
    marker = local_path / exist_check_file
    
    if marker.exists():
        logger.info(f"✅ Skipping download for {repo_id}, found {exist_check_file}.")
        return
    
    logger.info(f"⬇️  Downloading {repo_id} → {local_dir} …")
    
    # Create directory if it doesn't exist
    local_path.mkdir(parents=True, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_files_only=False  # fetch if not present
        )
        logger.info(f"🎉 Done downloading {repo_id} to {local_dir}.")
    except Exception as e:
        logger.error(f"❌ Failed to download {repo_id}: {e}")
        raise

# ──────────────────────────────────────────────────────────────────────────────
# 2) DOWNLOAD FUNCTIONS FOR EACH MODEL
# ──────────────────────────────────────────────────────────────────────────────

def download_blip_base():
    """Download base BLIP model."""
    safe_snapshot_download(
        repo_id="Salesforce/blip-image-captioning-base",
        local_dir="models/blip-image-captioning-base",
        exist_check_file="config.json"
    )

def download_llama2_7b():
    """Download Llama-2 7B Chat model."""
    safe_snapshot_download(
        repo_id="meta-llama/Llama-2-7b-chat-hf",
        local_dir="models/llama2-7b-chat",
        exist_check_file="config.json"
    )

def download_sdxl_base():
    """Download Stable Diffusion XL base model."""
    safe_snapshot_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        local_dir="models/sdxl-base",
        exist_check_file="model_index.json"
    )

# ──────────────────────────────────────────────────────────────────────────────
# 3) LOAD HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_blip_model(use_blip2: bool = False) -> Tuple[Any, Any]:
    """
    Load BLIP model for image analysis.
    
    Args:
        use_blip2: Whether to use BLIP-2 (more powerful) or base BLIP (lighter)
    
    Returns:
        Tuple of (processor, model)
    """
    if use_blip2:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        model_dir = "models/blip2-opt-2.7b"
        logger.info("Loading BLIP-2 from local cache...")
        
        processor = Blip2Processor.from_pretrained(model_dir, local_files_only=True)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )
    else:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        model_dir = "models/blip-image-captioning-base"
        logger.info("Loading BLIP base from local cache...")
        
        processor = BlipProcessor.from_pretrained(model_dir, local_files_only=True)
        model = BlipForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            local_files_only=True
        )
        
        device = get_device()
        model = model.to(device)
    
    model.eval()
    return processor, model

def load_llama_model(use_llama3: bool = False) -> Tuple[Any, Any]:
    """
    Load LLaMA model for text generation.
    
    Args:
        use_llama3: Whether to use Llama-3 8B or Llama-2 7B
    
    Returns:
        Tuple of (tokenizer, model)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    if use_llama3:
        model_dir = "models/llama3-8b-instruct"
        logger.info("Loading Llama-3 8B from local cache...")
    else:
        model_dir = "models/llama2-7b-chat"
        logger.info("Loading Llama-2 7B from local cache...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model

def load_sdxl_pipeline(use_turbo: bool = False) -> Any:
    """
    Load Stable Diffusion XL pipeline.
    
    Args:
        use_turbo: Whether to use SDXL Turbo (faster) or base SDXL (higher quality)
    
    Returns:
        Diffusion pipeline
    """
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    
    if use_turbo:
        model_dir = "models/sdxl-turbo"
        logger.info("Loading SDXL Turbo from local cache...")
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True
        )
    else:
        model_dir = "models/sdxl-base"
        logger.info("Loading SDXL base from local cache...")
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            local_files_only=True
        )
        
        # Use efficient scheduler for base model
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
    
    # Memory optimizations
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    
    device = get_device()
    if device.type == "cuda":
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except ImportError:
                logger.warning("xformers not available, skipping memory optimization")
    
    return pipe.to(device)

# ──────────────────────────────────────────────────────────────────────────────
# 4) UNIFIED DOWNLOAD AND LOAD FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def ensure_all_models_downloaded():
    """Download all required models if not already present."""
    logger.info("🔍 Checking for required models...")
    download_blip_base()
    download_llama2_7b()
    download_sdxl_base()
    logger.info("✅ All required models are ready!")

def load_all_models(use_larger_models: bool = False) -> dict:
    """
    Load all models for the pipeline.
    
    Args:
        use_larger_models: Whether to load larger, more powerful models
    
    Returns:
        Dictionary containing all loaded models
    """
    logger.info("🚀 Loading all models...")
    
    # Ensure models are downloaded first
    ensure_all_models_downloaded()
    
    models = {}
    
    try:
        # Load BLIP
        blip_processor, blip_model = load_blip_model(use_blip2=use_larger_models)
        models['blip'] = {'processor': blip_processor, 'model': blip_model}
        
        # Load LLaMA
        llama_tokenizer, llama_model = load_llama_model(use_llama3=use_larger_models)
        models['llama'] = {'tokenizer': llama_tokenizer, 'model': llama_model}
        
        # Load SDXL
        sdxl_pipeline = load_sdxl_pipeline(use_turbo=not use_larger_models)
        models['sdxl'] = {'pipeline': sdxl_pipeline}
        
        logger.info("🎉 All models loaded successfully!")
        return models
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise

# ──────────────────────────────────────────────────────────────────────────────
# 5) CONVENIENCE FUNCTIONS FOR PIPELINE INTEGRATION
# ──────────────────────────────────────────────────────────────────────────────

def get_model_paths() -> dict:
    """Get local paths for all models."""
    return {
        'blip': 'models/blip-image-captioning-base',
        'llama': 'models/llama2-7b-chat',
        'sdxl': 'models/sdxl-base'
    }

def check_models_exist(use_larger_models: bool = False) -> bool:
    """
    Check if all required models are already downloaded.
    
    Args:
        use_larger_models: Whether to check for larger model variants
    
    Returns:
        True if all models exist, False otherwise
    """
    paths = get_model_paths()
    
    check_files = {
        'blip': 'config.json',
        'llama': 'config.json',
        'sdxl': 'model_index.json'
    }
    
    for model_type, model_path in paths.items():
        check_file = Path(model_path) / check_files[model_type]
        if not check_file.exists():
            return False
    
    return True

# ──────────────────────────────────────────────────────────────────────────────
# 6) MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and test Storyweaver models")
    parser.add_argument("--large", action="store_true", help="Use larger, more powerful models")
    parser.add_argument("--check-only", action="store_true", help="Only check if models exist")
    parser.add_argument("--download-only", action="store_true", help="Only download, don't load")
    
    args = parser.parse_args()
    
    if args.check_only:
        exists = check_models_exist(args.large)
        print(f"Models exist: {exists}")
        exit(0 if exists else 1)
    
    if args.download_only:
        ensure_all_models_downloaded()
        print("Download complete!")
    else:
        # Download and load all models
        models = load_all_models(args.large)
        print("All models ready to go!")
        
        # Print model info
        print("\nLoaded models:")
        for model_type, model_dict in models.items():
            print(f"  {model_type}: {list(model_dict.keys())}") 