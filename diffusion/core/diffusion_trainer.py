from utils.checkpoint_compat import CheckpointCompatibilityMixin
import os
import math
from pathlib import Path
from typing import Dict, Optional, Union
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import re

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from torch.serialization import add_safe_globals
from deepspeed.runtime.fp16.loss_scaler import LossScaler

from .visualization import DiffusionVisualizer
from .sampler import DiffusionSampler

logger = get_logger(__name__)

add_safe_globals([LossScaler, TrainingArguments])

class DiffusionTrainer(CheckpointCompatibilityMixin, Trainer):
    """A trainer class for diffusion-based training of language models, extending HuggingFace Trainer."""

    def __init__(
        self,
        model=None,
        args: TrainingArguments = None,
        train_dataset=None,
        eval_dataset=None,
        visual_tokenizer=None,
        task_configs: Dict[str, Dict] = None,
        **kwargs
    ):
        """Initialize the trainer with model, datasets, and training arguments."""
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )


        self.setup_diffusion_config()


        self.task_configs = task_configs or {}


        self.sampler = DiffusionSampler(model, self.mask_token_id)

    def setup_diffusion_config(self):
        """Setup diffusion-specific configuration."""

        self.vocab_size = self.model.config.vocab_size
        self.mask_token_id = 126083  # Match the training token, not config.mask_token_id.
        logger.info(f"Using default vocab size: {self.vocab_size}")


    def forward_process(self, batch, prompt_lengths=None, eps=1e-2):
        """Apply forward diffusion process to input sequence."""
        input_ids = batch['input_ids'] if isinstance(batch, dict) else batch
        batch_size, seq_length = input_ids.shape
        device = input_ids.device


        t = torch.rand((batch_size,), device=device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask.unsqueeze(1).expand(-1, seq_length)


        while True:
            mask_indices = torch.rand((batch_size, seq_length), device=device) < p_mask
            if mask_indices.sum() > 0:
                break


        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)


        if prompt_lengths is not None:
            position_indices = torch.arange(seq_length, device=device).expand(batch_size, seq_length)
            prompt_mask = position_indices < prompt_lengths.unsqueeze(1)
            noisy_batch = torch.where(prompt_mask, input_ids, noisy_batch)
            mask_indices = mask_indices & ~prompt_mask

        return noisy_batch, mask_indices, p_mask

    def mask_diffusion_loss(self, logits, target, mask_indices, p_mask):
        'Compute the inverse-probability-weighted masked-token loss.'

        if not mask_indices.any():
            return logits.sum() * 0.0
        token_loss = F.cross_entropy(
            logits[mask_indices],
            target[mask_indices],
            reduction='none'
        ) / p_mask[mask_indices]


        return token_loss.mean()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        'Compute the diffusion training loss.'
        task_name = inputs.get('task_name', 'unknown')
        prompt_lengths = inputs.get('prompt_lengths')
        weights = inputs.get('weights', 1.0)
        if torch.is_tensor(weights):
            weights = weights.mean()


        task_config = self.task_configs.get(task_name, {}) if isinstance(task_name, str) else {}
        eps = task_config.get('eps', getattr(self.args, 'eps', 1e-2))
        loss_weight = task_config.get('loss_weight', 1.0)


        noisy_batch, mask_indices, p_mask = self.forward_process(
            inputs,
            prompt_lengths=prompt_lengths,
            eps=eps
        )


        outputs = model(noisy_batch)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs


        loss = self.mask_diffusion_loss(logits, inputs['input_ids'], mask_indices, p_mask)


        loss = loss * weights


        if num_items_in_batch is not None:
            loss = loss * (inputs['input_ids'].shape[0] / num_items_in_batch)

        torch.cuda.empty_cache()
        return (loss, outputs) if return_outputs else loss

    def evaluation_step(self, model, inputs):
        'Evaluate a batch without gradients.'
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
            if self.args.local_rank <= 0:
                task_name = inputs.get('task_name', 'unknown')
                logger.info(f"Eval loss for task {task_name}: {loss.item():.4f}")
        return loss

    def _save(self, output_dir: str, state_dict=None):
        """Save model checkpoint.

        Args:
            output_dir: The directory to save the checkpoint
            state_dict: The state dictionary to save (optional)
        """
        super()._save(output_dir, state_dict=state_dict)
        if self.is_world_process_zero():
            torch.save({
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
            }, os.path.join(output_dir, "meta.pt"))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Load from checkpoint."""
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        meta_path = os.path.join(resume_from_checkpoint, "meta.pt")
        if os.path.isfile(meta_path):
            meta = torch.load(meta_path)
            self.state.epoch = meta["epoch"]
            self.state.global_step = meta["global_step"]
