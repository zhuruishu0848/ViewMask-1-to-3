"""Generate three new views from one image, or four views from text alone."""
import argparse
import os
from pathlib import Path


DEFAULT_I2MV_QUESTION = (
    'Render the perspectives corresponding to 90, 180, and 270 degrees counterclockwise '
    'around the object, given the image and its appearance description.'
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('i2mv', 't2mv'), required=True)
    parser.add_argument('--model_path', default=os.environ.get('FINAL_CHECKPOINT', './checkpoints/viewmask1-to-3'), help='HF model or output directory; picks latest complete numbered checkpoint')
    parser.add_argument('--model_name_or_path', '--tokenizer_name_or_path', dest='model_name_or_path', default=os.environ.get('INFERENCE_TOKENIZER'), help='Tokenizer directory matching the model token IDs')
    parser.add_argument('--magvit_path', default=os.environ.get('MAGVIT_PATH', './checkpoints/magvitv2'))
    parser.add_argument('--image_path', help='Required for i2mv')
    parser.add_argument('--question', help='Optional view instruction for i2mv; required object description for t2mv')
    parser.add_argument('--output_dir', default='./outputs/inference')
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--mask_token_id', type=int, default=126083, help='Historical training code uses 126083, despite config.mask_token_id=126336')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default='bf16')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not args.model_path:
        parser.error('Set --model_path or FINAL_CHECKPOINT')
    if args.mode == 'i2mv' and (not args.image_path or not Path(args.image_path).is_file()):
        parser.error('i2mv requires an existing --image_path')
    if not args.question or not args.question.strip():
        if args.mode == 't2mv':
            parser.error('t2mv requires --question with an object description')
        args.question = DEFAULT_I2MV_QUESTION
    if args.num_steps < 1 or args.temperature <= 0:
        parser.error('num_steps and temperature must be positive')
    return args


def main():
    args = parse_args()
    import torch
    from PIL import Image
    from dataset import SPECIAL_TOKENS
    from utils.checkpoints import resolve_checkpoint
    from utils.inference_utils import setup_inference_environment, process_image, get_visual_tokens, diffusion_sample
    from utils.multiview import build_multiview_input, decode_multiview
    from utils.training import validate_tokenizer
    args.model_path = resolve_checkpoint(args.model_path)
    if not args.model_name_or_path:
        if not (Path(args.model_path) / 'tokenizer.json').is_file():
            raise ValueError('Checkpoint has no tokenizer files; set --tokenizer_name_or_path to the matching stage-2 checkpoint')
        args.model_name_or_path = args.model_path
    model, tokenizer, visual_tokenizer = setup_inference_environment(args)
    validate_tokenizer(tokenizer, model.config, list(SPECIAL_TOKENS.values()) + ['<|VISUAL_0|>', '<|VISUAL_8191|>'])
    visual_tokens = None
    images = []
    if args.mode == 'i2mv':
        image = process_image(args.image_path, 256).to(args.device)
        visual_tokens, _ = get_visual_tokens(visual_tokenizer, image)
        images.append(Image.fromarray((image[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')))
    ids, spans = build_multiview_input(tokenizer, args.question, visual_tokens, max_length=args.max_length, mask_token_id=args.mask_token_id, device=args.device)
    with torch.no_grad():
        generated = diffusion_sample(model, ids, args.mask_token_id, num_steps=args.num_steps, temperature=args.temperature, device=args.device)
        images.extend(decode_multiview(visual_tokenizer, tokenizer, generated, spans))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grid = Image.new('RGB', (256 * len(images), 256), 'white')
    for index, image in enumerate(images):
        image.save(output / f'view_{index:03d}.png')
        grid.paste(image, (256 * index, 0))
    grid.save(output / 'grid.png')
    print(f'Saved {len(images)} views to {output.resolve()}')


if __name__ == '__main__':
    main()
