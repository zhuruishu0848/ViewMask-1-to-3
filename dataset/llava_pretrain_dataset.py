import json
import random
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from dataset import SPECIAL_TOKENS


def encode_unicode_folder(name):
    return "".join(char if ord(char) < 128 else f"#U{ord(char):04x}" for char in str(name))


def safe_encode_unicode_folder(name):
    name = str(name)
    return Path(name) if "#U" in name else Path(encode_unicode_folder(name))


class LLaVAPretrainDataset(Dataset):
    def __init__(self, json_path, visual_tokenizer, text_tokenizer, image_size=256, max_length=1024):
        super().__init__()
        self.json_path = Path(json_path)
        self.image_root = self.json_path.parent
        self.visual_tokenizer = visual_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.device = next(visual_tokenizer.parameters()).device
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        print(f"Found {len(self.samples)} samples in LLaVA pretrain dataset")

    def __len__(self):
        return len(self.samples)

    def _load_and_process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def _encode_image_to_tokens(self, image):
        with torch.no_grad():
            token_ids = self.visual_tokenizer.get_code(image)
        return [f"<|VISUAL_{token_id}|>" for token_id in token_ids.flatten().cpu().numpy()]

    def _build_sequence(self, input_tokens, question, answer):
        user_prompt = (
            SPECIAL_TOKENS["MMU"]
            + SPECIAL_TOKENS["SOT"]
            + question
            + SPECIAL_TOKENS["EOT"]
            + SPECIAL_TOKENS["SOI"]
            + "".join(input_tokens)
            + SPECIAL_TOKENS["EOI"]
        )
        assistant_response = SPECIAL_TOKENS["SOT"] + answer + SPECIAL_TOKENS["EOT"]
        return user_prompt + assistant_response

    def _get_prompt_length(self, input_ids):
        eoi_token_id = self.text_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["EOI"])
        eoi_indices = (input_ids == eoi_token_id).nonzero(as_tuple=True)[0]
        return eoi_indices[-1].item() + 1 if len(eoi_indices) > 0 else 0

    def __getitem__(self, idx):
        for attempt in range(10):
            try:
                sample = self.samples[idx]
                image_path = self.image_root / sample["image"]
                image = self._load_and_process_image(image_path)
                question = sample["conversations"][0]["value"].replace("<image>", "").strip()
                answer = sample["conversations"][1]["value"]
                sequence = self._build_sequence(self._encode_image_to_tokens(image), question, answer)
                tokenized = self.text_tokenizer(
                    sequence,
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                    return_token_type_ids=False,
                    padding_side="right",
                )
                tokenized["input_ids"] = tokenized["input_ids"].squeeze(0)
                tokenized["attention_mask"] = tokenized["attention_mask"].squeeze(0)
                tokenized["prompt_lengths"] = self._get_prompt_length(tokenized["input_ids"])
                tokenized["task_name"] = "image_understanding"
                return tokenized
            except Exception as exc:
                print(f"Error loading sample {idx} (attempt {attempt + 1}/10): {exc}")
                if attempt == 9:
                    raise RuntimeError(f"Failed to fetch a valid sample after 10 retries for idx={idx}") from exc
                idx = random.randint(0, len(self) - 1)
