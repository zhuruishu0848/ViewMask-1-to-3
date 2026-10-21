"""Run a checkpoint on GSO i2mv and evaluate only the three generated views.

Invoke from the repository root: python -m eval.evaluate_gso_checkpoint --help
Metrics use generated views only, with torchvision InceptionV3 for FID.
"""
import argparse
import contextlib
import csv
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tokenizer_path', required=True)
    parser.add_argument('--magvit_path', required=True)
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--gso100_csv', required=True)
    parser.add_argument('--inception_weights', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--max_length', type=int, default=1536)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--limit', type=int, default=-1, help='Debug only; default uses the complete JSON')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    parser.add_argument('--metrics_only', action='store_true')
    return parser.parse_args()


def save_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def generate(args, samples, output):
    import numpy as np
    import torch
    from PIL import Image
    from dataset import SPECIAL_TOKENS
    from utils.inference_utils import setup_model_for_inference, setup_visual_tokenizer, process_image, diffusion_sample
    from utils.multiview import build_multiview_input, decode_multiview
    from utils.training import validate_tokenizer
    model, tokenizer, _ = setup_model_for_inference(args.model_path, args.tokenizer_path, args.device, args.precision)
    visual = setup_visual_tokenizer(args.magvit_path, args.device)
    validate_tokenizer(tokenizer, model.config, list(SPECIAL_TOKENS.values()) + ['<|VISUAL_0|>', '<|VISUAL_8191|>'])
    print('Model and MAGVIT loaded', flush=True)
    started = time.monotonic()
    completed = 0
    for offset in range(0, len(samples), args.batch_size):
        group = samples[offset:offset + args.batch_size]
        batch_number = offset // args.batch_size
        # Per-batch seeds make interrupted runs reproducible with the same batch size.
        torch.manual_seed(args.seed + batch_number)
        torch.cuda.manual_seed_all(args.seed + batch_number)
        if all((output / 'raw' / s['object_id'] / 'complete.json').is_file() for s in group):
            completed += len(group)
            continue
        tick = time.monotonic()
        with torch.inference_mode():
            pixels = torch.cat([process_image(s.get('source_image') or s['or_image'], 256) for s in group]).to(args.device)
            codes = visual.get_code(pixels)
            input_rows, all_spans = [], []
            for sample, code in zip(group, codes):
                tokens = [f'<|VISUAL_{n}|>' for n in code.flatten().tolist()]
                row, spans = build_multiview_input(tokenizer, sample['question'], tokens, max_length=args.max_length, device=args.device)
                input_rows.append(row)
                all_spans.append(spans)
            initial = torch.cat(input_rows)
            with contextlib.redirect_stdout(io.StringIO()):
                generated = diffusion_sample(model, initial, 126083, num_steps=args.num_steps, temperature=args.temperature, device=args.device)
            for index, (sample, spans) in enumerate(zip(group, all_spans)):
                generated_images = decode_multiview(visual, tokenizer, generated[index:index+1], spans)
                source = Image.fromarray((pixels[index].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8))
                images = [source] + generated_images
                folder = output / 'raw' / sample['object_id']
                folder.mkdir(parents=True, exist_ok=True)
                for view, image in enumerate(images):
                    image.save(folder / f'{view}.jpg')  # Original inference script's JPEG encoding.
                    image.save(folder / f'{view}.png')  # Also keep lossless images for inspection.
                grid = Image.new('RGB', (1024, 512), 'white')
                for view, image in enumerate(images):
                    grid.paste(image, (view * 256, 0))
                gt_paths = [sample.get('source_image') or sample['or_image']] + [sample[f'target_image_{angle}'] for angle in (90, 180, 270)]
                for view, path in enumerate(gt_paths):
                    with Image.open(path) as raw:
                        rgba = raw.convert('RGBA')
                        white = Image.new('RGB', rgba.size, 'white')
                        white.paste(rgba, mask=rgba.getchannel('A'))
                        grid.paste(white.resize((256, 256), Image.Resampling.BICUBIC), (view * 256, 256))
                grid.save(output / 'comparisons' / f'{offset+index:04d}_{sample["object_id"]}.png')
                save_json(folder / 'complete.json', {'object_id': sample['object_id'], 'sample_index': offset+index,
                                                    'batch_seed': args.seed + batch_number, 'generated_views': 3})
        torch.cuda.synchronize()
        completed += len(group)
        seconds = time.monotonic() - tick
        elapsed = time.monotonic() - started
        print(f'Generated {completed}/{len(samples)}; batch {seconds:.2f}s; elapsed {elapsed/60:.1f}min; peak CUDA {torch.cuda.max_memory_allocated()/2**30:.1f}GiB', flush=True)
        save_json(output / 'progress.json', {'phase': 'generation', 'completed_objects': completed,
                                            'total_objects': len(samples), 'elapsed_seconds': elapsed,
                                            'updated_utc': datetime.now(timezone.utc).isoformat()})
    del model, visual
    torch.cuda.empty_cache()


def evaluate(args, samples, output):
    import cv2
    import lpips
    import numpy as np
    import torch
    from PIL import Image
    from scipy import linalg
    from torchvision import models, transforms
    torch.set_num_threads(8)
    lpips_model = lpips.LPIPS(net='alex', verbose=False).to(args.device).eval()
    inception = models.inception_v3(weights=None, aux_logits=True, transform_input=False, init_weights=False)
    inception.load_state_dict(torch.load(args.inception_weights, map_location='cpu', weights_only=True))
    inception.fc = torch.nn.Identity()
    inception = inception.to(args.device).eval()
    to_tensor = transforms.ToTensor()
    feature_transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    allowed = {row[0] for row in csv.reader(Path(args.gso100_csv).open()) if row}

    def load(path):
        with Image.open(path) as raw:
            rgba = raw.convert('RGBA')
            image = Image.new('RGB', rgba.size, 'white')
            image.paste(rgba, mask=rgba.getchannel('A'))
            return image.resize((256, 256), Image.Resampling.LANCZOS)

    def ssim(a, b):
        a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
        b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
        mu1, mu2 = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
        sig1 = cv2.GaussianBlur(a*a, (11, 11), 1.5) - mu1*mu1
        sig2 = cv2.GaussianBlur(b*b, (11, 11), 1.5) - mu2*mu2
        sig12 = cv2.GaussianBlur(a*b, (11, 11), 1.5) - mu1*mu2
        c1, c2 = (0.01*255)**2, (0.03*255)**2
        return float((((2*mu1*mu2+c1)*(2*sig12+c2))/((mu1*mu1+mu2*mu2+c1)*(sig1+sig2+c2))).mean())

    rows, gen_features, gt_features, subset_mask = [], [], [], []
    for index, sample in enumerate(samples):
        folder = output / 'raw' / sample['object_id']
        if not (folder / 'complete.json').is_file():
            raise RuntimeError(f'Missing completed sample: {sample["object_id"]}')
        gen = [load(folder / f'{view}.jpg') for view in (1, 2, 3)]
        gt = [load(sample[f'target_image_{angle}']) for angle in (90, 180, 270)]
        with torch.inference_mode():
            g = torch.stack([to_tensor(im) for im in gen]).to(args.device)
            t = torch.stack([to_tensor(im) for im in gt]).to(args.device)
            perceptual = lpips_model(g*2-1, t*2-1).flatten().cpu().tolist()
            features = inception(torch.stack([feature_transform(im) for im in gen+gt]).to(args.device)).float().cpu().numpy()
        gen_features.extend(features[:3])
        gt_features.extend(features[3:])
        subset_mask.extend([sample['object_id'] in allowed]*3)
        for view, (im1, im2, distance) in enumerate(zip(gen, gt, perceptual), start=1):
            a, b = np.asarray(im1), np.asarray(im2)
            mse = np.mean((a.astype(np.float64)-b.astype(np.float64))**2)
            psnr = float(20*np.log10(255/np.sqrt(mse))) if mse else float('inf')
            rows.append({'object_id': sample['object_id'], 'view': view, 'gso100': sample['object_id'] in allowed,
                         'psnr': psnr, 'ssim': ssim(a, b), 'lpips': distance})
        if (index+1) % 25 == 0 or index+1 == len(samples):
            print(f'Metrics {index+1}/{len(samples)}', flush=True)
            save_json(output/'progress.json', {'phase':'metrics', 'completed_objects':index+1,'total_objects':len(samples)})
    with (output/'metrics_per_view.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    gen_features, gt_features = np.asarray(gen_features), np.asarray(gt_features)
    np.savez_compressed(output/'inception_features.npz', generated=gen_features, ground_truth=gt_features,
                        gso100=np.asarray(subset_mask))

    def summarize(mask):
        selected = [row for row, include in zip(rows, mask) if include]
        if not selected:
            return {'objects': 0, 'generated_views': 0, 'fid_torchvision_original_protocol': None}
        x, y = gt_features[mask], gen_features[mask]
        print(f'Computing original-protocol FID for {len(selected)} image pairs', flush=True)
        mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
        cov_x, cov_y = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
        covmean = linalg.sqrtm(cov_x.dot(cov_y))
        if np.iscomplexobj(covmean): covmean = covmean.real
        fid = float(np.sum((mean_x-mean_y)**2) + np.trace(cov_x+cov_y-2*covmean))
        result = {'objects':len(selected)//3, 'generated_views':len(selected)}
        for metric in ('psnr','ssim','lpips'):
            values = [row[metric] for row in selected]
            result[metric] = float(np.mean(values));result[metric+'_std'] = float(np.std(values,ddof=1))
        result['fid_torchvision_original_protocol'] = fid
        return result
    summary = {'checkpoint':args.model_path, 'protocol':{
        'mode':'i2mv', 'num_steps':args.num_steps, 'temperature':args.temperature, 'max_length':args.max_length,
        'seed':args.seed,'batch_size':args.batch_size,'image_size':256,'mask_token_id':126083,
        'paired_metrics':'Generated views 1/2/3 only; source view excluded; original JPEG save; grayscale Gaussian SSIM; LPIPS alex v0.1',
        'fid':'torchvision InceptionV3 with ImageNet normalization; not torch-fidelity FID'},
        'gso_all':summarize(np.ones(len(rows),dtype=bool)),
        'gso100':summarize(np.asarray(subset_mask,dtype=bool))}
    save_json(output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    save_json(output/'progress.json', {'phase':'complete','completed_objects':len(samples),'total_objects':len(samples)})


def main():
    args = parse_args()
    import torch
    from utils.checkpoints import resolve_checkpoint
    torch.set_num_threads(8)
    args.model_path = resolve_checkpoint(args.model_path)
    output = Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    (output/'comparisons').mkdir(exist_ok=True)
    samples = json.loads(Path(args.json_path).read_text())
    if args.limit > 0: samples = samples[:args.limit]
    if len({s['object_id'] for s in samples}) != len(samples):
        raise ValueError('Object IDs are not unique; refusing to overwrite samples')
    config = vars(args).copy(); config.pop('metrics_only')
    config['objects'] = len(samples)
    manifest = output/'run_config.json'
    save_json(manifest, config)
    print(f'Running {len(samples)} objects; checkpoint={args.model_path}; output={output}',flush=True)
    if not args.metrics_only: generate(args,samples,output)
    evaluate(args,samples,output)


if __name__ == '__main__':
    main()
