# Storyweaver

A machine learning pipeline for story generation that combines image analysis, text generation, and image synthesis.

## Project Structure

```
storyweaver/
├── README.md
├── requirements.txt          # pip libs persisted by AI Studio UI
├── test_model_download.py    # test model caching system
├── scripts/
│   └── download_models.py    # pre-download all models
├── notebooks/
│   ├── 01_extract_attributes.ipynb
│   ├── 02_generate_copy.ipynb
│   └── 03_generate_images.ipynb
├── src/
│   ├── __init__.py
│   ├── infer_blip.py
│   ├── infer_llama.py
│   ├── infer_sdxl.py
│   ├── pipeline.py           # orchestrates the three stages
│   ├── download_and_load_models.py  # intelligent model caching
│   └── utils.py
├── models/                   # auto-created model cache
│   ├── blip-image-captioning-base/
│   ├── llama2-7b-chat/
│   └── sdxl-base/
├── mlflow_runs/              # auto‑created by MLflow
├── demo/                     # copied into MLflow artifacts → Swagger‑hosted
│   ├── app.py                # Streamlit UI
│   └── assets/
│       └── placeholder.jpg
└── tests/
    └── smoke_test.py
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download models once (recommended):**
   ```bash
   python scripts/download_models.py
   ```
   This downloads ~22GB of models to local cache, avoiding repeated downloads.

3. Run the pipeline:
   ```bash
   python src/pipeline.py path/to/your/image.jpg
   ```

4. Launch the demo:
   ```bash
   streamlit run demo/app.py
   ```

## Pipeline Stages

1. **Extract Attributes** - Uses BLIP to analyze images and extract attributes
2. **Generate Copy** - Uses LLaMA to generate story text based on attributes  
3. **Generate Images** - Uses SDXL to create new images for the story

## Model Caching System

The pipeline includes an intelligent model caching system that:

- **Downloads models once**: Models are cached locally in the `models/` directory
- **Skips repeated downloads**: If models exist, they're loaded from cache instantly
- **Judge-friendly**: No manual uploads needed - just run the pipeline
- **Optional dataset mount**: Works with AI Studio datasets for persistent storage
- **Automatic fallback**: Falls back to HuggingFace download if cache is empty

### Models Used:
- **BLIP Image Captioning Base** (~2GB) - For image analysis
- **Llama-2 7B Chat** (~13GB) - For story generation  
- **Stable Diffusion XL Base** (~7GB) - For image generation
- **Total**: ~22GB cached locally

## Testing

Run smoke tests:
```bash
python -m pytest tests/
``` 