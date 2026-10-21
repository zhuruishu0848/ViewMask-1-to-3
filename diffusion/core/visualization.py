import os
from datetime import datetime
import re
import torch
import numpy as np
from PIL import Image

class DiffusionVisualizer:
    """Visualization utilities for diffusion model outputs."""
    
    def __init__(self, tokenizer, visual_tokenizer, output_dir="./outputs/validation"):
        self.tokenizer = tokenizer
        self.visual_tokenizer = visual_tokenizer
        self.output_dir = output_dir
        
    def visualize_batch(self, generated, input_ids):
        """Visualize a batch of generated samples alongside their ground truth."""
        pattern = r"<\|VISUAL_(\d+)\|>"
        
        for step, (single_ids, label_ids) in enumerate(zip(generated, input_ids)):
            # Decode tokens to text
            tokens = self.tokenizer.batch_decode([single_ids], skip_special_tokens=True)[0]
            label_tokens = self.tokenizer.batch_decode([label_ids], skip_special_tokens=True)[0]
            
            # Extract and process visual tokens
            generated_visual_ids = self._extract_visual_tokens(tokens, pattern)
            label_visual_ids = self._extract_visual_tokens(label_tokens, pattern)
            
            # Generate images
            images = self._decode_images(generated_visual_ids)
            label_images = self._decode_images(label_visual_ids)
            
            # Save results
            self._save_results(step, images, label_images, tokens, label_tokens)
            
    def _extract_visual_tokens(self, tokens, pattern):
        """Extract and process visual tokens from text."""
        ids = re.findall(pattern, tokens)
        visual_ids = torch.tensor([int(ids[i]) for i in range(256)]).unsqueeze(0)
        return torch.clamp(visual_ids, max=8191, min=0)
        
    def _decode_images(self, ids):
        """Decode image tokens to actual images."""
        with torch.no_grad():
            images = self.visual_tokenizer.decode_code(ids)
            
        images = torch.clamp(images, max=1.0, min=0.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return Image.fromarray(images.squeeze(0))
        
    def _save_results(self, step, images, label_images, tokens, label_tokens):
        """Save visualization results to disk."""
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = os.path.join(self.output_dir, current_time)
        os.makedirs(folder_path, exist_ok=True)
        
        # Save images
        images.save(f"{folder_path}/image_{step}.png")
        label_images.save(f"{folder_path}/label_{step}.png")
        
        # Save text
        with open(f"{folder_path}/text_{step}.txt", "w", encoding="utf-8") as f:
            f.write("prediction:\n")
            f.write(tokens)
            f.write("\nlabels:")
            f.write(label_tokens)
            
    @staticmethod
    def prepare_demo_batch(input_ids, prompt_lengths, mask_token_id):
        """Prepare demonstration batch for generation."""
        demo_init = input_ids.clone()
        position_indices = torch.arange(input_ids.size(1), device=input_ids.device)
        position_indices = position_indices.expand(input_ids.size(0), -1)
        
        completion_mask = position_indices >= prompt_lengths.unsqueeze(1)
        demo_init[completion_mask] = mask_token_id
        
        return demo_init 