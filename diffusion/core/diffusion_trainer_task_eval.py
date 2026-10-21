from utils.checkpoint_compat import CheckpointCompatibilityMixin
import os
import math
from pathlib import Path
from typing import Dict, Optional, Union
from datetime import datetime
from collections import defaultdict
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

class DiffusionTrainerTask_Eval(CheckpointCompatibilityMixin, Trainer):
    """A trainer class for diffusion-based training of language models, extending HuggingFace Trainer."""

    def __init__(self, model=None, args: TrainingArguments=None, train_dataset=None, eval_dataset=None, visual_tokenizer=None, task_configs: Dict[str, Dict]=None, eval_frequency: int=500, **kwargs):
        """Initialize the trainer with model, datasets, and training arguments."""
        super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, **kwargs)
        self.setup_diffusion_config()
        self.task_configs = task_configs or {}
        self.sampler = DiffusionSampler(model, self.mask_token_id)
        self.task_losses = defaultdict(list)
        self.task_loss_tracker = defaultdict(list)
        self.task_sample_counter = defaultdict(int)
        self.loss_log_frequency = getattr(args, 'loss_log_frequency', eval_frequency)
        self.eval_frequency = eval_frequency
        self.eval_task_losses = defaultdict(list)
        self.last_eval_step = -1

    def setup_diffusion_config(self):
        """Setup diffusion-specific configuration."""
        self.vocab_size = self.model.config.vocab_size
        self.mask_token_id = 126083  # Match the token used during training.
        logger.info(f'Using default vocab size: {self.vocab_size}')

    def forward_process(self, batch, prompt_lengths=None, eps=0.01, is_eval=False):
        """Apply forward diffusion process to input sequence."""
        input_ids = batch['input_ids'] if isinstance(batch, dict) else batch
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        if is_eval and prompt_lengths is not None:
            position_indices = torch.arange(seq_length, device=device).expand(batch_size, seq_length)
            mask_indices = position_indices >= prompt_lengths.unsqueeze(1)
            noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
            for batch_idx in range(batch_size):
                eoi_pos = prompt_lengths[batch_idx].item()
                special_positions = [eoi_pos, eoi_pos + 257, eoi_pos + 257 + 1, eoi_pos + 257 + 1 + 257, eoi_pos + 257 + 1 + 257 + 1, eoi_pos + 257 + 1 + 257 + 1 + 257]
                for pos in special_positions:
                    if pos < seq_length:
                        noisy_batch[batch_idx, pos] = input_ids[batch_idx, pos]
                        mask_indices[batch_idx, pos] = False
            p_mask = torch.ones((batch_size, seq_length), device=device)
        else:
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
        return (noisy_batch, mask_indices, p_mask)

    def mask_diffusion_loss(self, logits, target, mask_indices, p_mask, task_names, weights, is_eval=False):
        'Compute masked-token loss and task averages.'
        batch_size, seq_len = target.shape
        if not mask_indices.any():
            logger.warning('No masked positions found in batch!')
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        token_loss = F.cross_entropy(logits[mask_indices], target[mask_indices], reduction='none') / (p_mask[mask_indices] + 1e-08)
        sample_indices = torch.arange(batch_size, device=target.device).unsqueeze(1).expand(-1, seq_len)
        masked_sample_indices = sample_indices[mask_indices]
        task_sample_groups = {}
        task_weights = {}
        for i, task_name in enumerate(task_names):
            if task_name not in task_sample_groups:
                task_sample_groups[task_name] = []
                task_weights[task_name] = 0.0
            task_sample_groups[task_name].append(i)
            task_weights[task_name] += weights[i]
        for task_name in task_weights:
            task_weights[task_name] /= len(task_sample_groups[task_name])
        task_losses = {}
        total_weight = sum(task_weights.values())
        if total_weight == 0:
            logger.warning('Total weight is zero!')
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        final_loss = 0.0
        for task_name, sample_indices_list in task_sample_groups.items():
            task_mask = torch.isin(masked_sample_indices, torch.tensor(sample_indices_list, device=target.device))
            if task_mask.any():
                task_loss = token_loss[task_mask].mean()
                task_losses[task_name] = task_loss.item()
                normalized_weight = task_weights[task_name] / total_weight
                final_loss += normalized_weight * task_loss
            else:
                task_losses[task_name] = 0.0
        loss_dict = self.eval_task_losses if is_eval else self.task_losses
        for task_name, loss_val in task_losses.items():
            loss_dict[task_name].append(loss_val)
            if not is_eval:
                self.task_sample_counter[task_name] += len(task_sample_groups[task_name])
        return final_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        'Compute training or generation-evaluation loss.'
        task_names = inputs.get('task_name')
        prompt_lengths = inputs.get('prompt_lengths')
        weights = inputs.get('weights')
        is_eval = inputs.get('is_eval', False)
        if isinstance(task_names, str):
            task_names = [task_names]
        if isinstance(weights, (int, float)):
            weights = [weights]
        if is_eval:
            num_steps = getattr(self.args, 'eval_num_steps', 50)
            temperature = getattr(self.args, 'eval_temperature', 0.8)
            initial_ids, _, _ = self.forward_process(inputs, prompt_lengths=prompt_lengths, eps=getattr(self.args, 'eps', 0.01), is_eval=True)
            generated_ids = self.iterative_diffusion_sample(model=model, initial_ids=initial_ids, mask_token_id=self.mask_token_id, num_steps=num_steps, temperature=temperature)
            outputs = model(generated_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            batch_size, seq_length = inputs['input_ids'].shape
            device = inputs['input_ids'].device
            position_indices = torch.arange(seq_length, device=device).expand(batch_size, seq_length)
            eval_mask_indices = position_indices >= prompt_lengths.unsqueeze(1)
            for batch_idx in range(batch_size):
                eoi_pos = prompt_lengths[batch_idx].item()
                special_positions = self._get_special_token_positions(eoi_pos, seq_length)
                for pos in special_positions:
                    if 0 <= pos < seq_length:
                        eval_mask_indices[batch_idx, pos] = False
            p_mask = torch.ones((batch_size, seq_length), device=device)
            loss = self.mask_diffusion_loss(logits=logits, target=inputs['input_ids'], mask_indices=eval_mask_indices, p_mask=p_mask, task_names=task_names, weights=weights, is_eval=True)
        else:
            noisy_batch, mask_indices, p_mask = self.forward_process(inputs, prompt_lengths=prompt_lengths, eps=getattr(self.args, 'eps', 0.01), is_eval=False)
            outputs = model(noisy_batch)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = self.mask_diffusion_loss(logits, inputs['input_ids'], mask_indices, p_mask, task_names, weights, is_eval=False)
        torch.cuda.empty_cache()
        return (loss, outputs) if return_outputs else loss

    def _get_special_token_positions(self, eoi_pos, seq_length):
        IMAGE_TOKEN_LENGTH = 257
        special_positions = []
        for img_idx in range(3):
            offset = img_idx * (IMAGE_TOKEN_LENGTH + 1)
            soi_pos = eoi_pos + offset
            eoi_pos_img = soi_pos + IMAGE_TOKEN_LENGTH
            if 0 <= soi_pos < seq_length:
                special_positions.append(soi_pos)
            if 0 <= eoi_pos_img < seq_length:
                special_positions.append(eoi_pos_img)
        return special_positions

    def iterative_diffusion_sample(self, model, initial_ids, mask_token_id, num_steps=50, temperature=0.8):
        'Sample masked tokens through iterative denoising.'
        model.eval()
        batch_size, seq_length = initial_ids.shape
        device = initial_ids.device
        ids = initial_ids.clone()
        mask_indices = ids == mask_token_id
        if not mask_indices.any():
            return ids
        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)[:-1]
        prev_p_mask = 1.0
        with torch.no_grad():
            for i, t in enumerate(ts):
                p_mask = t.item()
                if i > 0:
                    keep_mask = torch.rand(mask_indices.shape, device=device) < p_mask / prev_p_mask
                    mask_indices = mask_indices & keep_mask
                x_t = torch.where(mask_indices, mask_token_id, ids)
                outputs = model(x_t)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                if mask_indices.any():
                    masked_logits = logits[mask_indices]
                    probs = torch.nn.functional.softmax(masked_logits / temperature, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    ids[mask_indices] = next_token_ids
                prev_p_mask = p_mask
        return ids

    def training_step(self, model, inputs, num_items_in_batch=None):
        'Run a training step and periodic generation evaluation.'
        loss = super().training_step(model, inputs, num_items_in_batch)
        if (self.state.global_step + 1) % self.loss_log_frequency == 0 and self.is_world_process_zero():
            self._log_task_losses()
        if self.eval_dataset is not None and (self.state.global_step + 1) % self.eval_frequency == 0 and (self.state.global_step != self.last_eval_step) :
            self._run_evaluation()
            self.last_eval_step = self.state.global_step
        return loss

    def _run_evaluation(self):
        logger.info('=' * 60)
        logger.info(f'Running Evaluation at Step {self.state.global_step}')
        logger.info('=' * 60)
        model = self.model_wrapped
        model.eval()
        self.eval_task_losses.clear()
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        total_eval_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for i, inputs in enumerate(eval_dataloader):
                inputs = self._prepare_inputs(inputs)
                inputs['is_eval'] = True
                loss = self.compute_loss(model, inputs)
                total_eval_loss += loss.item()
                total_samples += 1
        avg_eval_loss = total_eval_loss / total_samples if total_samples > 0 else 0.0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            totals = torch.tensor([total_eval_loss, total_samples], device=self.args.device, dtype=torch.float64)
            torch.distributed.all_reduce(totals)
            avg_eval_loss = (totals[0] / totals[1].clamp_min(1)).item()
        if self.is_world_process_zero():
            self._log_eval_results(avg_eval_loss)
        model.train()
        logger.info(f'Evaluation completed. Average loss: {avg_eval_loss:.6f}')
        logger.info('=' * 60)

    def _log_eval_results(self, avg_eval_loss):
        'Log evaluation metrics.'
        eval_log_dict = {}
        logger.info('-' * 60)
        logger.info('Evaluation Task Loss Statistics:')
        logger.info('-' * 60)
        for task_name, losses in self.eval_task_losses.items():
            if losses:
                task_avg_loss = np.mean(losses)
                logger.info(f'{task_name:20s} | Eval Loss: {task_avg_loss:.6f} | Samples: {len(losses):6d}')
                eval_log_dict[f'eval_loss/{task_name}'] = task_avg_loss
        eval_log_dict['eval_loss/gso_100'] = avg_eval_loss
        logger.info('-' * 60)
        logger.info(f"{'Overall':20s} | Eval Loss: {avg_eval_loss:.6f}")
        logger.info('-' * 60)
        if hasattr(self, 'log') and eval_log_dict:
            import logging
            tb_logger = logging.getLogger('transformers.trainer')
            original_level = tb_logger.level
            tb_logger.setLevel(logging.ERROR)
            self.log(eval_log_dict)
            tb_logger.setLevel(original_level)

    def _log_task_losses(self):
        'Log and reset task loss statistics.'
        if not self.task_losses:
            return
        simple_log_dict = {}
        logger.info('=' * 60)
        logger.info(f'Task Loss Statistics (Step {self.state.global_step})')
        logger.info('=' * 60)
        total_samples = 0
        overall_loss = 0
        for task_name, losses in self.task_losses.items():
            if losses:
                avg_loss = np.mean(losses)
                min_loss = np.min(losses)
                max_loss = np.max(losses)
                num_samples = len(losses)
                logger.info(f'{task_name:20s} | Samples: {num_samples:6d} | Avg: {avg_loss:.6f} | Min: {min_loss:.6f} | Max: {max_loss:.6f}')
                simple_log_dict[f'train_loss/{task_name}'] = avg_loss
                total_samples += num_samples
                overall_loss += avg_loss * num_samples
        if total_samples > 0:
            overall_avg = overall_loss / total_samples
            logger.info('-' * 60)
            logger.info(f"{'Overall':20s} | Samples: {total_samples:6d} | Avg: {overall_avg:.6f}")
            simple_log_dict['train_loss/overall'] = overall_avg
        logger.info('=' * 60)
        if hasattr(self, 'log') and simple_log_dict:
            import logging
            tb_logger = logging.getLogger('transformers.trainer')
            original_level = tb_logger.level
            tb_logger.setLevel(logging.ERROR)
            self.log(simple_log_dict)
            tb_logger.setLevel(original_level)
        self.task_losses.clear()

    def evaluation_step(self, model, inputs):
        'Evaluate a batch without gradients.'
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
            if self.args.local_rank <= 0:
                task_name = inputs.get('task_name', 'unknown')
                logger.info(f'Eval loss for task {task_name}: {loss.item():.4f}')
        return loss

    def _save(self, output_dir: str, state_dict=None):
        """Save model checkpoint."""
        super()._save(output_dir, state_dict=state_dict)
        if self.is_world_process_zero():
            torch.save({'epoch': self.state.epoch, 'global_step': self.state.global_step, 'last_eval_step': self.last_eval_step}, os.path.join(output_dir, 'meta.pt'))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Load from checkpoint."""
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        meta_path = os.path.join(resume_from_checkpoint, 'meta.pt')
        if os.path.isfile(meta_path):
            meta = torch.load(meta_path)
            self.state.epoch = meta['epoch']
            self.state.global_step = meta['global_step']
            self.last_eval_step = meta.get('last_eval_step', -1)
