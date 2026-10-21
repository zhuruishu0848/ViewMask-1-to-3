import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from transformers import AutoConfig, AutoModel, AutoTokenizer

from tokenizer.modeling_magvitv2 import MAGVITv2


WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.pt",
    "model.pth",
    "model.bin",
    "checkpoint.pt",
)


def process_image(image_path, image_size, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    image = Image.open(image_path)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    else:
        image = image.convert("RGB")
    return transform(image).unsqueeze(0)


def _schedule_fraction(step, num_steps, schedule_type, device):
    progress = (step + 1) / num_steps
    if schedule_type == "cosine":
        return torch.cos(torch.tensor(progress * np.pi / 2, device=device))
    if schedule_type == "linear":
        return 1.0 - progress
    if schedule_type == "quadratic":
        return 1.0 - progress**2
    raise ValueError(f"Unknown schedule_type: {schedule_type}")


@torch.no_grad()
def diffusion_sample(model, initial_ids, mask_token_id, num_steps=18, temperature=1.0, device="cuda", schedule_type="cosine"):
    model.eval()
    ids = initial_ids.to(device)
    mask_indices = ids == mask_token_id

    num_masked = mask_indices.sum().item()
    print(f"Masked tokens: {num_masked}")
    if num_masked == 0:
        return ids

    total_masked = mask_indices.sum(dim=1, keepdim=True)
    for step in range(num_steps):
        current_masked = mask_indices.sum().item()
        if current_masked == 0:
            break

        keep_n = (
            (total_masked.float() * _schedule_fraction(step, num_steps, schedule_type, device)).floor().clamp_min(1).long()
            if step < num_steps - 1
            else torch.zeros_like(total_masked)
        )

        outputs = model(ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        masked_logits = logits[mask_indices]
        probs = F.softmax(masked_logits / temperature, dim=-1)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs).clamp(min=1e-10)))
        sampled_ids = (masked_logits / temperature + gumbel_noise).argmax(dim=-1)
        confidence = probs.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)
        ids[mask_indices] = sampled_ids

        new_mask_indices = torch.zeros_like(mask_indices)
        offset = 0
        for batch_idx in range(ids.shape[0]):
            batch_positions = mask_indices[batch_idx].nonzero(as_tuple=False).squeeze(-1)
            batch_count = batch_positions.numel()
            batch_keep = int(keep_n[batch_idx].item())
            if batch_count == 0 or batch_keep <= 0:
                offset += batch_count
                continue

            batch_confidence = confidence[offset:offset + batch_count]
            offset += batch_count
            batch_keep = min(batch_keep, batch_count)
            scores = batch_confidence + torch.randn_like(batch_confidence) * temperature if temperature > 0 else batch_confidence
            selected = torch.topk(scores, k=batch_keep, largest=False, sorted=False).indices
            new_mask_indices[batch_idx, batch_positions[selected]] = True

        ids[new_mask_indices] = mask_token_id
        mask_indices = new_mask_indices

    return ids


@torch.no_grad()
def diffusion_sample_with_cfg(model, conditional_ids, mask_token_id, cfg_scale=4.0, num_steps=50, temperature=0.8, device="cuda"):
    if cfg_scale == 1.0:
        return diffusion_sample(model, conditional_ids, mask_token_id, num_steps, temperature, device)

    model.eval()
    conditional_ids = conditional_ids.to(device)
    unconditional_ids = conditional_ids.clone()
    sot_token_id = 134543
    eot_token_id = 134544

    sot_positions = (conditional_ids == sot_token_id).nonzero(as_tuple=True)
    eot_positions = (conditional_ids == eot_token_id).nonzero(as_tuple=True)
    if len(sot_positions[1]) and len(eot_positions[1]):
        sot_pos = sot_positions[1][0].item()
        eot_pos = eot_positions[1][0].item()
        if eot_pos > sot_pos + 1:
            unconditional_ids = torch.full_like(conditional_ids, mask_token_id)
            unconditional_ids[0, :sot_pos + 1] = conditional_ids[0, :sot_pos + 1]
            unconditional_ids[0, sot_pos + 1] = eot_token_id
            remaining = conditional_ids[0, eot_pos + 1:]
            end_pos = min(sot_pos + 2 + remaining.numel(), conditional_ids.shape[1])
            unconditional_ids[0, sot_pos + 2:end_pos] = remaining[:end_pos - sot_pos - 2]

    ids = torch.cat([unconditional_ids, conditional_ids], dim=0)
    cond_mask_positions = (conditional_ids == mask_token_id).nonzero(as_tuple=True)[1]
    uncond_mask_positions = (unconditional_ids == mask_token_id).nonzero(as_tuple=True)[1]
    num_masks = min(cond_mask_positions.numel(), uncond_mask_positions.numel())
    if num_masks == 0:
        return conditional_ids

    cond_mask_positions = cond_mask_positions[:num_masks]
    uncond_mask_positions = uncond_mask_positions[:num_masks]
    active = torch.ones(num_masks, dtype=torch.bool, device=device)
    prev_p_mask = 1.0

    for step, t in enumerate(torch.linspace(1.0, 0.0, num_steps + 1, device=device)[:-1]):
        p_mask = t.item()
        if step > 0:
            active &= torch.rand_like(active, dtype=torch.float32) < p_mask / prev_p_mask
        if not active.any():
            break

        x_t = ids.clone()
        cond_positions = cond_mask_positions[active]
        uncond_positions = uncond_mask_positions[active]
        x_t[1, cond_positions] = mask_token_id
        x_t[0, uncond_positions] = mask_token_id

        outputs = model(x_t)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        guided_logits = logits[0, uncond_positions] + cfg_scale * (logits[1, cond_positions] - logits[0, uncond_positions])
        probs = F.softmax(guided_logits / temperature, dim=-1)
        sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        ids[1, cond_positions] = sampled_ids
        ids[0, uncond_positions] = sampled_ids
        prev_p_mask = p_mask

    return ids[1:2]


def _dtype_for_precision(precision):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _load_checkpoint_state(model_path, device):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    index_file = model_path / "model.safetensors.index.json"
    if index_file.is_file():
        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        state_dict = {}
        for shard_file in sorted(set(index["weight_map"].values())):
            state_dict.update(load_file(str(model_path / shard_file), device=device))
        return state_dict

    if model_path.is_dir():
        for weight_file in WEIGHT_FILES:
            weight_path = model_path / weight_file
            if weight_path.is_file():
                model_path = weight_path
                break
        else:
            raise FileNotFoundError(f"No supported weight file found in {model_path}")

    if model_path.suffix == ".safetensors":
        return load_file(str(model_path), device=device)
    return torch.load(model_path, map_location=device)


def setup_model_for_inference(model_path, model_name_or_path, device, precision):
    """Load checkpoint config/weights together; tokenizer may live in a separate directory."""
    from utils.checkpoints import resolve_checkpoint
    dtype = _dtype_for_precision(precision)
    model_dir = Path(model_path)
    if model_dir.is_dir() and ((model_dir / "config.json").is_file() or any(model_dir.glob("checkpoint-*"))):
        model_source = resolve_checkpoint(model_path)
        tokenizer_source = model_name_or_path or model_source
        config = AutoConfig.from_pretrained(model_source, trust_remote_code=True, torch_dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_source, config=config, trust_remote_code=True, torch_dtype=dtype)
    else:
        # Compatibility with older callers that pass a standalone weight file.
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True, torch_dtype=dtype)
        if model_path != model_name_or_path:
            state_dict = _load_checkpoint_state(model_path, "cpu")
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            elif isinstance(state_dict, dict) and "module" in state_dict:
                state_dict = state_dict["module"]
            model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model, tokenizer, config


def setup_visual_tokenizer(magvit_path, device):
    visual_tokenizer = MAGVITv2.from_pretrained(magvit_path).to(device)
    visual_tokenizer.eval()
    visual_tokenizer.requires_grad_(False)
    return visual_tokenizer


def save_images(visual_tokenizer, token_ids, output_dir, prefix="generated"):
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        images = visual_tokenizer.decode_code(token_ids)

    images = torch.clamp(images, min=0.0, max=1.0) * 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    for idx, image_array in enumerate(images):
        Image.fromarray(image_array).save(os.path.join(output_dir, f"{prefix}_{idx}.png"))


def get_visual_tokens(visual_tokenizer, image):
    with torch.no_grad():
        visual_token_ids = visual_tokenizer.get_code(image)
    tokens = [f"<|VISUAL_{token_id}|>" for token_id in visual_token_ids.flatten().cpu().numpy()]
    return tokens, visual_token_ids


def prepare_tokenized_prompt(tokenizer, prompt_str, max_length, mask_token_id):
    tokenized_prompt = tokenizer(
        prompt_str,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
        padding_side="right",
    )
    return tokenized_prompt.input_ids


def extract_visual_ids_from_text(text, device):
    visual_ids = [int(token_id) for token_id in re.findall(r"<\|VISUAL_(\d+)\|>", text)]
    if not visual_ids:
        return torch.empty((1, 0), dtype=torch.long, device=device)
    return torch.tensor(visual_ids, dtype=torch.long, device=device).unsqueeze(0).clamp(min=0, max=8191)


def setup_inference_environment(args):
    seed = getattr(args, "seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, tokenizer, _ = setup_model_for_inference(
        args.model_path,
        args.model_name_or_path,
        args.device,
        args.precision,
    )
    visual_tokenizer = setup_visual_tokenizer(
        getattr(args, "magvit_path", os.environ.get("MAGVIT_PATH", "showlab/magvitv2")),
        args.device,
    )

    if hasattr(args, "output_dir"):
        os.makedirs(args.output_dir, exist_ok=True)

    return model, tokenizer, visual_tokenizer
