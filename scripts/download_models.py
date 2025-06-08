#!/usr/bin/env python3
"""
Standalone script to download all Storyweaver models.
Run this once to cache models locally and avoid repeated downloads.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from download_and_load_models import ensure_all_models_downloaded, check_models_exist, get_model_paths


def main():
    """Download all models and verify they exist."""
    print("🚀 Storyweaver Model Downloader")
    print("=" * 50)
    
    # Check if models already exist
    if check_models_exist():
        print("✅ All models already downloaded and cached!")
        print("\nModel locations:")
        paths = get_model_paths()
        for model_type, path in paths.items():
            print(f"  {model_type}: {path}")
        return
    
    print("📥 Downloading models (this may take several minutes)...")
    print("\nThis will download:")
    print("  • BLIP Image Captioning Base (~2GB)")
    print("  • Llama-2 7B Chat (~13GB)")
    print("  • Stable Diffusion XL Base (~7GB)")
    print("  Total: ~22GB")
    print()
    
    try:
        # Download all models
        ensure_all_models_downloaded()
        
        print("\n🎉 All models downloaded successfully!")
        print("\nModel locations:")
        paths = get_model_paths()
        for model_type, path in paths.items():
            size = get_directory_size(path)
            print(f"  {model_type}: {path} ({size:.1f}GB)")
        
        print(f"\nTotal disk usage: {get_total_model_size():.1f}GB")
        print("\n✨ You can now run the Storyweaver pipeline without internet access!")
        
    except Exception as e:
        print(f"\n❌ Error downloading models: {e}")
        return 1
    
    return 0


def get_directory_size(path: str) -> float:
    """Get directory size in GB."""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size / (1024**3)  # Convert to GB
    except:
        return 0.0


def get_total_model_size() -> float:
    """Get total size of all models in GB."""
    paths = get_model_paths()
    total = 0
    for path in paths.values():
        total += get_directory_size(path)
    return total


if __name__ == "__main__":
    exit(main()) 