# Storyweaver

A machine learning pipeline for story generation that combines image analysis, text generation, and image synthesis.

## Project Structure

```
storyweaver/
├── README.md
├── requirements.txt          # pip libs persisted by AI Studio UI
├── notebooks/
│   ├── 01_extract_attributes.ipynb
│   ├── 02_generate_copy.ipynb
│   └── 03_generate_images.ipynb
├── src/
│   ├── infer_blip.py
│   ├── infer_llama.py
│   ├── infer_sdxl.py
│   ├── pipeline.py           # orchestrates the three stages
│   └── utils.py
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

2. Run the pipeline:
   ```bash
   python src/pipeline.py
   ```

3. Launch the demo:
   ```bash
   streamlit run demo/app.py
   ```

## Pipeline Stages

1. **Extract Attributes** - Uses BLIP to analyze images and extract attributes
2. **Generate Copy** - Uses LLaMA to generate story text based on attributes  
3. **Generate Images** - Uses SDXL to create new images for the story

## Testing

Run smoke tests:
```bash
python -m pytest tests/
``` 