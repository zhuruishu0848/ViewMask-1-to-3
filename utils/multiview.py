"""Shared i2mv/t2mv prompt layout, matching the training sequence."""
from dataset import SPECIAL_TOKENS


def build_multiview_input(tokenizer, question, visual_tokens=None, *, max_length=2048, mask_token_id=126083, device='cpu'):
    import torch
    prompt = SPECIAL_TOKENS['I2MI'] + SPECIAL_TOKENS['SOT'] + question + SPECIAL_TOKENS['EOT']
    if visual_tokens is not None:
        if len(visual_tokens) != 256:
            raise ValueError('This checkpoint expects 256 visual tokens per 256x256 image')
        prompt += SPECIAL_TOKENS['SOI'] + ''.join(visual_tokens) + SPECIAL_TOKENS['EOI']
    # Match dataset tokenization (including tokenizer BOS behavior).
    prompt_ids = tokenizer(prompt, return_token_type_ids=False)['input_ids']
    count = 4 if visual_tokens is None else 3
    if len(prompt_ids) + count * 258 > max_length:
        raise ValueError(f'Prompt and {count} image blocks need {len(prompt_ids) + count * 258} tokens; increase --max_length')
    ids = torch.full((1, max_length), mask_token_id, dtype=torch.long, device=device)
    ids[0, :len(prompt_ids)] = torch.tensor(prompt_ids, device=device)
    spans = []
    for i in range(count):
        start = len(prompt_ids) + i * 258
        ids[0, start] = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['SOI'])
        ids[0, start + 257] = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['EOI'])
        spans.append((start + 1, start + 257))
    return ids, spans


def decode_multiview(visual_tokenizer, tokenizer, ids, spans):
    import re
    import torch
    from PIL import Image
    images = []
    for start, end in spans:
        tokens = tokenizer.convert_ids_to_tokens(ids[0, start:end].tolist())
        matches = [re.fullmatch(r'<\|VISUAL_(\d+)\|>', token) for token in tokens]
        if not all(matches):
            raise ValueError('Generated non-visual tokens in an image block; check model/tokenizer compatibility and sampling settings')
        codes = torch.tensor([[int(match[1]) for match in matches]], device=ids.device)
        with torch.no_grad():
            pixels = visual_tokenizer.decode_code(codes).clamp(0, 1)
        array = (pixels[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype('uint8')
        images.append(Image.fromarray(array))
    return images
