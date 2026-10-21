from utils.checkpoint_compat import CheckpointCompatibilityMixin
import os
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from torch.serialization import add_safe_globals
from transformers import Trainer, TrainingArguments

from .sampler import DiffusionSampler

logger = get_logger(__name__)
add_safe_globals([LossScaler, TrainingArguments])


class DiffusionTrainerTask(CheckpointCompatibilityMixin, Trainer):
    def __init__(
        self,
        model=None,
        args: TrainingArguments = None,
        train_dataset=None,
        eval_dataset=None,
        visual_tokenizer=None,
        task_configs: Dict[str, Dict] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )
        self.visual_tokenizer = visual_tokenizer
        self.task_configs = task_configs or {}
        self.vocab_size = self.model.config.vocab_size
        self.mask_token_id = 126083
        self.sampler = DiffusionSampler(model, self.mask_token_id)
        self.task_losses = defaultdict(list)
        self.task_sample_counter = defaultdict(int)
        self.loss_log_frequency = getattr(args, "loss_log_frequency", 100)
        logger.info("Using vocab size: %s", self.vocab_size)

    def forward_process(self, batch, prompt_lengths=None, eps=1e-2):
        input_ids = batch["input_ids"] if isinstance(batch, dict) else batch
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        t = torch.rand((batch_size,), device=device)
        p_mask = ((1 - eps) * t + eps).unsqueeze(1).expand(-1, seq_length)

        while True:
            mask_indices = torch.rand((batch_size, seq_length), device=device) < p_mask
            if mask_indices.any():
                break

        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        if prompt_lengths is not None:
            positions = torch.arange(seq_length, device=device).expand(batch_size, seq_length)
            prompt_mask = positions < prompt_lengths.unsqueeze(1)
            noisy_batch = torch.where(prompt_mask, input_ids, noisy_batch)
            mask_indices = mask_indices & ~prompt_mask

        return noisy_batch, mask_indices, p_mask

    def _normalize_metadata(self, task_names, weights, batch_size, device):
        if isinstance(task_names, str):
            task_names = [task_names]
        elif task_names is None:
            task_names = ["unknown"] * batch_size

        if weights is None:
            weights = torch.ones(batch_size, device=device)
        elif not torch.is_tensor(weights):
            weights = torch.tensor(weights if isinstance(weights, list) else [weights], device=device, dtype=torch.float32)
        else:
            weights = weights.to(device=device, dtype=torch.float32)

        if weights.ndim == 0:
            weights = weights.repeat(batch_size)
        if len(task_names) != batch_size:
            task_names = list(task_names)[:batch_size] + ["unknown"] * max(0, batch_size - len(task_names))
        if weights.numel() != batch_size:
            weights = torch.ones(batch_size, device=device)

        return list(task_names), weights

    def mask_diffusion_loss(self, logits, target, mask_indices, p_mask, task_names, weights):
        batch_size, seq_length = target.shape
        device = target.device
        task_names, weights = self._normalize_metadata(task_names, weights, batch_size, device)

        token_loss = F.cross_entropy(
            logits[mask_indices],
            target[mask_indices],
            reduction="none",
        ) / p_mask[mask_indices]

        sample_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_length)
        masked_sample_indices = sample_indices[mask_indices]

        task_groups = defaultdict(list)
        for sample_idx, task_name in enumerate(task_names):
            task_groups[task_name].append(sample_idx)

        task_weights = {
            task_name: weights[indices].mean()
            for task_name, indices in task_groups.items()
        }
        total_weight = sum(task_weights.values())
        if total_weight.item() == 0:
            total_weight = torch.ones((), device=device)

        final_loss = logits.new_tensor(0.0)
        for task_name, indices in task_groups.items():
            task_mask = torch.isin(masked_sample_indices, torch.tensor(indices, device=device))
            if task_mask.any():
                task_loss = token_loss[task_mask].mean()
                final_loss = final_loss + (task_weights[task_name] / total_weight) * task_loss
                self.task_losses[task_name].append(task_loss.detach().float().item())
            else:
                self.task_losses[task_name].append(0.0)
            self.task_sample_counter[task_name] += len(indices)

        return final_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_lengths = inputs.get("prompt_lengths")
        noisy_batch, mask_indices, p_mask = self.forward_process(
            inputs,
            prompt_lengths=prompt_lengths,
            eps=getattr(self.args, "eps", 1e-2),
        )
        outputs = model(noisy_batch)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = self.mask_diffusion_loss(
            logits,
            inputs["input_ids"],
            mask_indices,
            p_mask,
            inputs.get("task_name"),
            inputs.get("weights"),
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if (self.state.global_step + 1) % self.loss_log_frequency == 0 and self.is_world_process_zero():
            self._log_task_losses()
        return loss

    def _log_task_losses(self):
        if not self.task_losses:
            return

        logger.info("=" * 60)
        logger.info("Task Loss Statistics (Step %s)", self.state.global_step)
        logger.info("=" * 60)

        total_samples = 0
        weighted_loss = 0.0
        for task_name, losses in self.task_losses.items():
            if not losses:
                continue

            avg_loss = float(np.mean(losses))
            min_loss = float(np.min(losses))
            max_loss = float(np.max(losses))
            num_samples = len(losses)
            logger.info(
                "%20s | Samples: %6d | Avg: %.6f | Min: %.6f | Max: %.6f",
                task_name,
                num_samples,
                avg_loss,
                min_loss,
                max_loss,
            )
            self.log({
                f"train_loss/{task_name}": avg_loss,
                f"train_samples/{task_name}": num_samples,
                f"train_loss_min/{task_name}": min_loss,
                f"train_loss_max/{task_name}": max_loss,
            })
            total_samples += num_samples
            weighted_loss += avg_loss * num_samples

        if total_samples:
            overall_avg = weighted_loss / total_samples
            logger.info("%20s | Samples: %6d | Avg: %.6f", "Overall", total_samples, overall_avg)
            self.log({
                "train_loss/overall": overall_avg,
                "train_samples/overall": total_samples,
            })

        logger.info("=" * 60)
        self.task_losses.clear()

    def evaluation_step(self, model, inputs):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
            if self.args.local_rank <= 0:
                logger.info("Eval loss for task %s: %.4f", inputs.get("task_name", "unknown"), loss.item())
        return loss

    def _save(self, output_dir: str, state_dict=None):
        super()._save(output_dir, state_dict=state_dict)
        if self.is_world_process_zero():
            torch.save({
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
            }, os.path.join(output_dir, "meta.pt"))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        meta_path = os.path.join(resume_from_checkpoint, "meta.pt")
        if os.path.isfile(meta_path):
            meta = torch.load(meta_path)
            self.state.epoch = meta["epoch"]
            self.state.global_step = meta["global_step"]
