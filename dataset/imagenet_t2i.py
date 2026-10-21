import json
import os
import random
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset import SPECIAL_TOKENS

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def image_processor(images, image_area):
    if images.mode != "RGB":
        images = images.convert("RGB")
    resolution = int(image_area ** 0.5)
    image = transforms.Resize([resolution, resolution])(images)
    return {"pixel_values": transforms.ToTensor()(image)}


class ImageNetTokenizedDataset(Dataset):
    def __init__(
        self,
        root_dir,
        visual_tokenizer,
        text_tokenizer,
        phase="train",
        image_size=256,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=42,
        max_length=1024,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.visual_tokenizer = visual_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.device = next(visual_tokenizer.parameters()).device
        self.phase = phase
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.samples = self._load_dataset()
        print(f"Found {len(self.samples)} samples for {phase} phase")

    def _load_dataset(self):
        caption_file = self.root_dir / f"{self.phase}.json"
        with open(caption_file, "r", encoding="utf-8") as f:
            captions = json.load(f)

        image_dir = self.root_dir / self.phase
        samples = []
        for image_id, caption in tqdm(captions.items(), desc=f"Loading ImageNet {self.phase}"):
            class_id = image_id.split("_")[0]
            samples.append({
                "image_path": image_dir / class_id / f"{image_id}.JPEG",
                "caption": caption,
            })
        return samples

    def _load_and_process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def _encode_image_to_tokens(self, image):
        with torch.no_grad():
            token_ids = self.visual_tokenizer.get_code(image)
        return [f"<|VISUAL_{token_id}|>" for token_id in token_ids.flatten().cpu().numpy()]

    def _build_sequence(self, input_tokens, caption):
        tokens = (
            [SPECIAL_TOKENS["T2I"], SPECIAL_TOKENS["SOT"], caption, SPECIAL_TOKENS["EOT"], SPECIAL_TOKENS["SOI"]]
            + input_tokens
            + [SPECIAL_TOKENS["EOI"]]
        )
        return "".join(tokens)

    def _get_prompt_length(self, input_ids):
        eot_token_id = self.text_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["EOT"])
        eot_indices = (input_ids == eot_token_id).nonzero(as_tuple=True)[0]
        return eot_indices[0].item() + 1 if len(eot_indices) > 0 else 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, retry=10):
        for attempt in range(retry):
            try:
                sample = self.samples[idx]
                image = self._load_and_process_image(sample["image_path"])
                sequence = self._build_sequence(self._encode_image_to_tokens(image), sample["caption"])
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
                return tokenized
            except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
                print(f"Error loading sample {idx} (attempt {attempt + 1}/{retry}): {exc}")
                if attempt == retry - 1:
                    raise RuntimeError(f"Failed to fetch a valid sample after {retry} retries for idx={idx}") from exc
                idx = random.randint(0, len(self) - 1)
