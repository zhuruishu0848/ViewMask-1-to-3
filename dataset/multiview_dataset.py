import bisect
import json
import random
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset import SPECIAL_TOKENS


class MultiViewObjaverseDataset(Dataset):
    def __init__(self, json_file, visual_tokenizer, text_tokenizer, phase="train", image_size=256, max_length=2048):
        super().__init__()
        self.json_file = Path(json_file)
        self.visual_tokenizer = visual_tokenizer
        self.text_tokenizer = text_tokenizer
        self.phase = phase
        self.image_size = image_size
        self.max_length = max_length
        self.device = next(visual_tokenizer.parameters()).device
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.samples = self._load_samples()
        print(f"Found {len(self.samples)} samples for {phase} phase")

    def _load_samples(self):
        if not self.json_file.exists():
            raise FileNotFoundError(f"Sample file not found: {self.json_file}")

        with open(self.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for item in tqdm(data, desc=f"Processing {self.phase} samples"):
            samples.append({
                "source_image": self._image_path(item.get("source_image") or item["or_image"]),
                "caption": item.get("caption", False),
                "question": item["question"],
                "target_images": [
                    self._image_path(item["target_image_90"]),
                    self._image_path(item["target_image_180"]),
                    self._image_path(item["target_image_270"]),
                ],
                "category": item.get("category"),
                "object_id": item.get("object_id"),
            })
        return samples

    def _image_path(self, value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.json_file.parent / path

    def _load_and_process_image(self, image_path):
        # Match the original PNG preprocessing: composite alpha onto white.
        with Image.open(image_path) as raw:
            if raw.mode == "RGBA" or "transparency" in raw.info:
                image = raw.convert("RGBA")
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.getchannel("A"))
                image = background
            else:
                image = raw.convert("RGB")
            return self.transform(image).unsqueeze(0).to(self.device)

    def _encode_image_to_tokens(self, image):
        with torch.no_grad():
            token_ids = self.visual_tokenizer.get_code(image)
        return [f"<|VISUAL_{token_id}|>" for token_id in token_ids.flatten().cpu().numpy()]

    def _build_sequence(self, input_tokens, question, target_tokens_list):
        tokens = (
            [SPECIAL_TOKENS["I2MI"]]
            + [SPECIAL_TOKENS["SOT"], question, SPECIAL_TOKENS["EOT"]]
            + [SPECIAL_TOKENS["SOI"]] + input_tokens + [SPECIAL_TOKENS["EOI"]]
        )
        for target_tokens in target_tokens_list:
            tokens.extend([SPECIAL_TOKENS["SOI"]] + target_tokens + [SPECIAL_TOKENS["EOI"]])
        return "".join(tokens)

    def _get_prompt_length(self, input_ids, caption=False):
        eot_token_id = self.text_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["EOI"])
        eot_indices = (input_ids == eot_token_id).nonzero(as_tuple=True)[0]
        return eot_indices[0].item() + 1 if len(eot_indices) > 0 else 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_image = self._load_and_process_image(sample["source_image"])
        target_images = [self._load_and_process_image(path) for path in sample["target_images"]]
        sequence = self._build_sequence(
            self._encode_image_to_tokens(input_image),
            sample["question"],
            [self._encode_image_to_tokens(image) for image in target_images],
        )
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
        tokenized["prompt_lengths"] = self._get_prompt_length(tokenized["input_ids"], sample["caption"])
        tokenized["task_name"] = "image_multiview_generation"
        return tokenized


class MultiViewObjaverseCaptionDataset(MultiViewObjaverseDataset):
    """Use EOT for text conditioning and the first EOI for image conditioning.

    The question already contains the caption from the sample metadata.
    """

    def __init__(self, *args, text_condition_probability=0.5, **kwargs):
        if not 0 <= text_condition_probability <= 1:
            raise ValueError("text_condition_probability must be between 0 and 1")
        self.text_condition_probability = text_condition_probability
        super().__init__(*args, **kwargs)

    def _get_prompt_length(self, input_ids, caption=False):
        has_caption = bool(caption) and str(caption).lower() not in ("false", "none", "0")
        if has_caption and random.random() < self.text_condition_probability:
            token_id = self.text_tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["EOT"])
            positions = (input_ids == token_id).nonzero(as_tuple=True)[0]
            return positions[0].item() + 1 if len(positions) else 0
        return super()._get_prompt_length(input_ids)


class MultiViewDatasetCollection(Dataset):
    def __init__(self, datasets: dict):
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
        self.lengths = [len(datasets[name]) for name in self.dataset_names]
        self.cum_lengths = [0]
        for length in self.lengths:
            self.cum_lengths.append(self.cum_lengths[-1] + length)
        self.length = self.cum_lengths[-1]

        print("\nMultiview dataset:")
        for name, length in zip(self.dataset_names, self.lengths):
            print(f"Dataset: {name} - samples: {length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cum_lengths, idx) - 1
        dataset_name = self.dataset_names[dataset_idx]
        idx_in_dataset = idx - self.cum_lengths[dataset_idx]
        assert 0 <= idx_in_dataset < len(self.datasets[dataset_name]), (
            f"idx_in_dataset {idx_in_dataset} out of range for dataset {dataset_name} "
            f"with length {len(self.datasets[dataset_name])}, idx={idx}, cum_lengths={self.cum_lengths}"
        )

        item = self.datasets[dataset_name][idx_in_dataset]
        if item is None:
            raise RuntimeError(f"{dataset_name} dataset returned None for idx {idx_in_dataset}")
        item["weights"] = 1.0
        return item
