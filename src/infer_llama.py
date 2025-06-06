"""
LLaMA model inference for story text generation.
"""

from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from .utils import get_device, setup_logging

logger = setup_logging()


class LlamaInferencer:
    """LLaMA model for generating story text from image attributes."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """Initialize LLaMA model and tokenizer."""
        self.device = get_device()
        logger.info(f"Loading LLaMA model {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.generation_config = GenerationConfig(
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
    
    def create_story_prompt(self, attributes: Dict[str, Any]) -> str:
        """Create a prompt for story generation based on image attributes."""
        caption = attributes.get('caption', '')
        
        prompt_parts = [
            "Write a creative short story based on the following image description and attributes:",
            f"Image Description: {caption}",
            ""
        ]
        
        # Add other attributes
        for key, value in attributes.items():
            if key != 'caption' and value:
                prompt_parts.append(f"{key}: {value}")
        
        prompt_parts.extend([
            "",
            "Story Requirements:",
            "- Write in an engaging, narrative style",
            "- Include vivid descriptions",
            "- Create interesting characters",
            "- Develop a compelling plot",
            "- Keep it between 200-400 words",
            "",
            "Story:"
        ])
        
        return "\n".join(prompt_parts)
    
    def generate_story(self, attributes: Dict[str, Any]) -> str:
        """Generate a story based on image attributes."""
        prompt = self.create_story_prompt(attributes)
        
        # Format as chat if using chat model
        if "chat" in self.model.config.name_or_path.lower():
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        else:
            formatted_prompt = prompt
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )
        
        # Decode and clean up the response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the story part (remove the prompt)
        if "Story:" in generated_text:
            story = generated_text.split("Story:")[-1].strip()
        else:
            # Fallback: try to extract from chat format
            if "[/INST]" in generated_text:
                story = generated_text.split("[/INST]")[-1].strip()
            else:
                story = generated_text[len(formatted_prompt):].strip()
        
        logger.info(f"Generated story of {len(story.split())} words")
        return story
    
    def generate_multiple_variants(self, attributes: Dict[str, Any], num_variants: int = 3) -> List[str]:
        """Generate multiple story variants based on the same attributes."""
        stories = []
        for i in range(num_variants):
            logger.info(f"Generating story variant {i+1}/{num_variants}")
            story = self.generate_story(attributes)
            stories.append(story)
        return stories
    
    def refine_story(self, story: str, refinement_prompt: str) -> str:
        """Refine an existing story based on additional instructions."""
        prompt = f"""
        Original Story:
        {story}
        
        Refinement Instructions:
        {refinement_prompt}
        
        Refined Story:
        """
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        refined_story = generated_text.split("Refined Story:")[-1].strip()
        
        logger.info("Story refined successfully")
        return refined_story 