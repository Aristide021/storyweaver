"""
Main pipeline orchestrator for the storyweaver project.
Coordinates the three stages: attribute extraction, story generation, and image generation.
"""

import os
import json
from typing import Dict, Any, List, Optional
from PIL import Image
import mlflow
import mlflow.pytorch

from .infer_blip import BLIPInferencer
from .infer_llama import LlamaInferencer
from .infer_sdxl import SDXLInferencer
from .utils import load_image, save_results, setup_logging, validate_inputs
from .download_and_load_models import ensure_all_models_downloaded

logger = setup_logging()


class StoryweaverPipeline:
    """Main pipeline that orchestrates the three inference stages."""
    
    def __init__(
        self,
        use_cached_models: bool = True,
        blip_model: str = None,
        llama_model: str = None,
        sdxl_model: str = None,
        mlflow_experiment: str = "storyweaver"
    ):
        """Initialize the pipeline with model configurations."""
        self.mlflow_experiment = mlflow_experiment
        
        # Setup MLflow
        mlflow.set_experiment(mlflow_experiment)
        
        logger.info("Initializing Storyweaver Pipeline...")
        
        # Ensure models are downloaded if using cached models
        if use_cached_models:
            logger.info("Ensuring all models are downloaded and cached...")
            ensure_all_models_downloaded()
        
        # Initialize models (will use cached versions if model paths are None)
        self.blip = BLIPInferencer(blip_model)
        self.llama = LlamaInferencer(llama_model)
        self.sdxl = SDXLInferencer(sdxl_model)
        
        logger.info("Pipeline initialization complete!")
    
    def run_full_pipeline(
        self,
        input_image_path: str,
        output_dir: str = "outputs",
        num_story_variants: int = 1,
        num_generated_images: int = 3,
        custom_questions: Optional[List[str]] = None,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """Run the complete pipeline from input image to generated story and images."""
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("input_image", input_image_path)
            mlflow.log_param("num_story_variants", num_story_variants)
            mlflow.log_param("num_generated_images", num_generated_images)
            
            try:
                # Stage 1: Extract attributes from input image
                logger.info("Stage 1: Extracting attributes from input image...")
                input_image = load_image(input_image_path)
                attributes = self.blip.extract_attributes(input_image, custom_questions)
                
                mlflow.log_dict(attributes, "extracted_attributes.json")
                
                # Stage 2: Generate story text
                logger.info("Stage 2: Generating story text...")
                if num_story_variants == 1:
                    stories = [self.llama.generate_story(attributes)]
                else:
                    stories = self.llama.generate_multiple_variants(attributes, num_story_variants)
                
                # Use the first (or best) story for image generation
                main_story = stories[0]
                
                mlflow.log_text(main_story, "generated_story.txt")
                
                # Stage 3: Generate new images
                logger.info("Stage 3: Generating new images...")
                generated_images = self.sdxl.generate_story_images(
                    main_story,
                    attributes,
                    num_generated_images,
                    **generation_kwargs
                )
                
                # Save results
                results = self._save_results(
                    input_image_path,
                    attributes,
                    stories,
                    generated_images,
                    output_dir
                )
                
                # Log artifacts to MLflow
                self._log_mlflow_artifacts(results, output_dir)
                
                # Log metrics
                mlflow.log_metric("num_attributes", len(attributes))
                mlflow.log_metric("story_length", len(main_story.split()))
                mlflow.log_metric("num_generated_images", len(generated_images))
                
                logger.info(f"Pipeline completed successfully! Results saved to {output_dir}")
                return results
                
            except Exception as e:
                logger.error(f"Pipeline failed: {e}")
                mlflow.log_param("error", str(e))
                raise
    
    def run_stage_by_stage(
        self,
        input_image_path: str,
        output_dir: str = "outputs"
    ) -> Dict[str, Any]:
        """Run pipeline with intermediate saves for debugging."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Stage 1
        logger.info("Running Stage 1: Attribute Extraction")
        input_image = load_image(input_image_path)
        attributes = self.blip.extract_attributes(input_image)
        
        stage1_path = os.path.join(output_dir, "stage1_attributes.json")
        save_results(attributes, stage1_path)
        logger.info(f"Stage 1 complete. Attributes saved to {stage1_path}")
        
        # Stage 2
        logger.info("Running Stage 2: Story Generation")
        story = self.llama.generate_story(attributes)
        
        stage2_path = os.path.join(output_dir, "stage2_story.txt")
        with open(stage2_path, 'w') as f:
            f.write(story)
        logger.info(f"Stage 2 complete. Story saved to {stage2_path}")
        
        # Stage 3
        logger.info("Running Stage 3: Image Generation")
        generated_images = self.sdxl.generate_story_images(story, attributes)
        
        stage3_dir = os.path.join(output_dir, "stage3_images")
        os.makedirs(stage3_dir, exist_ok=True)
        
        image_paths = []
        for i, img in enumerate(generated_images):
            img_path = os.path.join(stage3_dir, f"generated_image_{i+1}.png")
            img.save(img_path)
            image_paths.append(img_path)
        
        logger.info(f"Stage 3 complete. Images saved to {stage3_dir}")
        
        return {
            "input_image": input_image_path,
            "attributes": attributes,
            "story": story,
            "generated_images": image_paths,
            "stage_outputs": {
                "stage1": stage1_path,
                "stage2": stage2_path,
                "stage3": stage3_dir
            }
        }
    
    def _save_results(
        self,
        input_image_path: str,
        attributes: Dict[str, Any],
        stories: List[str],
        generated_images: List[Image.Image],
        output_dir: str
    ) -> Dict[str, Any]:
        """Save all pipeline results to output directory."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save input image copy
        input_image = load_image(input_image_path)
        input_copy_path = os.path.join(output_dir, "input_image.jpg")
        input_image.save(input_copy_path)
        
        # Save generated images
        image_paths = []
        for i, img in enumerate(generated_images):
            img_path = os.path.join(output_dir, f"generated_image_{i+1}.png")
            img.save(img_path)
            image_paths.append(img_path)
        
        # Save text outputs
        attributes_path = os.path.join(output_dir, "attributes.json")
        save_results(attributes, attributes_path)
        
        story_path = os.path.join(output_dir, "story.txt")
        with open(story_path, 'w') as f:
            f.write(stories[0])  # Save main story
        
        # Save all stories if multiple variants
        if len(stories) > 1:
            for i, story in enumerate(stories):
                variant_path = os.path.join(output_dir, f"story_variant_{i+1}.txt")
                with open(variant_path, 'w') as f:
                    f.write(story)
        
        # Create summary
        summary = {
            "input_image": input_copy_path,
            "attributes": attributes_path,
            "main_story": story_path,
            "generated_images": image_paths,
            "pipeline_config": {
                "blip_model": self.blip.processor.name_or_path,
                "llama_model": self.llama.model.config.name_or_path,
                "sdxl_model": "stabilityai/stable-diffusion-xl-base-1.0"
            }
        }
        
        summary_path = os.path.join(output_dir, "pipeline_summary.json")
        save_results(summary, summary_path)
        
        return summary
    
    def _log_mlflow_artifacts(self, results: Dict[str, Any], output_dir: str):
        """Log artifacts to MLflow."""
        try:
            # Log the entire output directory
            mlflow.log_artifacts(output_dir, "pipeline_outputs")
            
            # Log input image
            mlflow.log_artifact(results["input_image"], "input")
            
            # Log generated images
            for img_path in results["generated_images"]:
                mlflow.log_artifact(img_path, "generated_images")
                
        except Exception as e:
            logger.warning(f"Failed to log MLflow artifacts: {e}")


def main():
    """Example usage of the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Storyweaver pipeline")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--stage-by-stage", action="store_true", help="Run with intermediate saves")
    parser.add_argument("--num-stories", type=int, default=1, help="Number of story variants")
    parser.add_argument("--num-images", type=int, default=3, help="Number of images to generate")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = StoryweaverPipeline()
    
    # Run pipeline
    if args.stage_by_stage:
        results = pipeline.run_stage_by_stage(args.input_image, args.output_dir)
    else:
        results = pipeline.run_full_pipeline(
            args.input_image,
            args.output_dir,
            args.num_stories,
            args.num_images
        )
    
    print(f"Pipeline completed! Results: {results}")


if __name__ == "__main__":
    main() 