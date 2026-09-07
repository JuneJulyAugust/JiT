"""Small pretrained JiT-B sampling demo; no ImageNet, DDP, or FID dependencies."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model', choices=['JiT-B/16', 'JiT-B/32'], default='JiT-B/16')
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    parser.add_argument('--class-ids', type=int, nargs='+', default=[207])
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--cfg', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/jit-b-demo'))
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}; see README_EVALUATION.md')
    if args.steps < 1:
        parser.error('--steps must be positive')
    if any(label < 0 or label >= 1000 for label in args.class_ids):
        parser.error('--class-ids must be ImageNet class indices in [0, 999]')
    # if args.output_dir.exists() and any(args.output_dir.iterdir()):
        # parser.error('--output-dir must be empty or new (existing results are never overwritten)')

    # The model has compile decorators; eager execution keeps a tiny demo portable.
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    import torch
    from PIL import Image
    from denoiser import Denoiser

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available; use --device mps or --device cpu')
    if device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is not available; use --device cpu')

    img_size = 256 if args.model == 'JiT-B/16' else 512
    config = SimpleNamespace(
        model=args.model, img_size=img_size, class_num=1000,
        attn_dropout=0.0, proj_dropout=0.0, label_drop_prob=0.1,
        P_mean=-0.8, P_std=0.8, t_eps=0.05,
        noise_scale=1.0 if img_size == 256 else 2.0,
        ema_decay1=0.9999, ema_decay2=0.9996,
        sampling_method='heun', num_sampling_steps=args.steps,
        cfg=args.cfg, interval_min=0.1, interval_max=1.0,
    )
    torch.manual_seed(args.seed)
    print(f'Loading {args.model} EMA1 weights from {args.checkpoint}', flush=True)
    # Do not fall back to unsafe pickle loading or randomly initialized weights.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if 'model_ema1' not in checkpoint:
        parser.error('Expected an official checkpoint containing model_ema1')
    model = Denoiser(config)
    model.load_state_dict(checkpoint['model_ema1'], strict=True)
    del checkpoint
    model = model.to(device).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device: {device}; {args.steps} Heun steps; batch size 1', flush=True)
    started = time.perf_counter()
    with torch.inference_mode():
        for index, label in enumerate(args.class_ids):
            labels = torch.tensor([label], device=device, dtype=torch.long)
            precision = (torch.autocast('cuda', dtype=torch.bfloat16)
                         if device == 'cuda' and torch.cuda.is_bf16_supported()
                         else nullcontext())
            with precision:
                pixels = model.generate(labels)
            if not torch.isfinite(pixels).all():
                raise RuntimeError('Sampling produced non-finite pixels')
            pixels = ((pixels[0].float().cpu() + 1) * 127.5).clamp(0, 255).round()
            pixels = pixels.to(torch.uint8).permute(1, 2, 0).numpy()
            output = args.output_dir / f'{index:03d}-class-{label:04d}.png'
            Image.fromarray(pixels).save(output)
            print(f'Saved {output} ({time.perf_counter() - started:.1f}s elapsed)', flush=True)

    metadata = vars(args).copy()
    metadata.update(device=device, img_size=img_size, weights='model_ema1',
                    sampling_method='heun', interval=[0.1, 1.0],
                    elapsed_seconds=time.perf_counter() - started,
                    torch_version=torch.__version__, metrics_computed=False)
    (args.output_dir / 'run.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    print('Sampling demo complete. This is not a 50,000-image FID/IS evaluation.')


if __name__ == '__main__':
    main()
