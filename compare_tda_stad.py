"""
Head-to-head comparison: TDA (ours) vs STAD-vMF (Schirmer et al.)

Runs both methods on the same dataset / stream and produces a
side-by-side results CSV + optional accuracy-over-time plot.
"""

import os
import csv
import json
import time
import random
import argparse
from datetime import datetime

import torch
import clip

from utils import (
    resolve_device,
    get_config_file,
    build_test_data_loader,
    clip_classifier,
)
from tda_runner import run_test_tda
from stad_baseline import run_stad


def get_arguments():
    parser = argparse.ArgumentParser(description='TDA vs STAD-vMF comparison')
    parser.add_argument('--config', required=True, help='Config directory')
    parser.add_argument('--datasets', required=True, help='Datasets separated by /')
    parser.add_argument('--data-root', default='./dataset/')
    parser.add_argument('--backbone', default='RN50', choices=['RN50', 'ViT-B/16'])
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--output-dir', default='results/comparison')

    # TDA SSM settings
    parser.add_argument('--tda-memory-type', default='ssm', choices=['cache', 'ssm'])
    parser.add_argument('--tda-ssm-mode', default='heuristic',
                        choices=['heuristic', 'vmf-fixed', 'vmf-adaptive', 'vmf-online'])

    # STAD settings
    parser.add_argument('--stad-kappa-trans', type=float, default=1000.0)
    parser.add_argument('--stad-kappa-ems', type=float, default=1000.0)
    parser.add_argument('--stad-window-size', type=int, default=3)
    parser.add_argument('--stad-n-em-iters', type=int, default=5)
    parser.add_argument('--stad-batch-sizes', type=str, default='1,16,64',
                        help='Comma-separated batch sizes for STAD')
    parser.add_argument('--stad-learn-kappa', action='store_true')

    return parser.parse_args()


def main():
    args = get_arguments()
    device = resolve_device(args.device)
    clip_model, preprocess = clip.load(args.backbone, device=device)
    if device.type == 'mps':
        clip_model.float()
    clip_model.eval()

    random.seed(1)
    torch.manual_seed(1)

    stad_batch_sizes = [int(x) for x in args.stad_batch_sizes.split(',')]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for dataset_name in args.datasets.split('/'):
        print(f"\n{'='*60}")
        print(f"  Comparing TDA vs STAD-vMF on: {dataset_name}")
        print(f"{'='*60}")

        cfg = get_config_file(args.config, dataset_name)
        test_loader, classnames, template = build_test_data_loader(
            dataset_name, args.data_root, preprocess
        )
        clip_weights = clip_classifier(classnames, template, clip_model, device=device)

        out_dir = os.path.join(args.output_dir, f"{dataset_name}_{timestamp}")
        os.makedirs(out_dir, exist_ok=True)

        all_results = []

        # ── TDA (ours) ──────────────────────────────────────────
        # Run with the original cache method
        print("\n--- TDA (cache) ---")
        random.seed(1); torch.manual_seed(1)
        tda_cache_result = run_test_tda(
            cfg['positive'], cfg['negative'],
            test_loader, clip_model, clip_weights,
            memory_type='cache',
            max_samples=args.max_samples,
            return_details=True,
        )
        all_results.append({
            'method': 'TDA-cache',
            **{k: v for k, v in tda_cache_result.items()
               if k not in ('cumulative_accuracy', 'forgetting_curve')},
        })
        print(f"  Accuracy: {tda_cache_result['accuracy']:.2f}%")

        # Run with SSM (heuristic – our best)
        print(f"\n--- TDA SSM ({args.tda_ssm_mode}) ---")
        ssm_kwargs = {
            'correction_mode': args.tda_ssm_mode,
            'kalman_q': 0.01,  'kalman_r': 0.05,
            'kalman_q_min': 0.005, 'kalman_q_max': 0.05,
            'kalman_r_min': 0.01,  'kalman_r_max': 0.1,
        }
        random.seed(1); torch.manual_seed(1)
        tda_ssm_result = run_test_tda(
            cfg['positive'], cfg['negative'],
            test_loader, clip_model, clip_weights,
            memory_type='ssm',
            max_samples=args.max_samples,
            return_details=True,
            ssm_kwargs=ssm_kwargs,
        )
        all_results.append({
            'method': f'TDA-SSM-{args.tda_ssm_mode}',
            **{k: v for k, v in tda_ssm_result.items()
               if k not in ('cumulative_accuracy', 'forgetting_curve')},
        })
        print(f"  Accuracy: {tda_ssm_result['accuracy']:.2f}%")

        # ── CLIP zero-shot baseline ─────────────────────────────
        print("\n--- CLIP zero-shot ---")
        random.seed(1); torch.manual_seed(1)
        zs_correct = []
        sample_count = 0
        for images, target in test_loader:
            if args.max_samples is not None and sample_count >= args.max_samples:
                break
            with torch.no_grad():
                images = images.to(device)
                feats = clip_model.encode_image(images).float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
                logits = 100.0 * feats @ clip_weights
                pred = logits.argmax(dim=1).item()
                zs_correct.append(float(pred == target.item()))
            sample_count += 1
        zs_acc = 100.0 * sum(zs_correct) / len(zs_correct) if zs_correct else 0.0
        all_results.append({
            'method': 'CLIP-zero-shot',
            'accuracy': zs_acc,
            'num_samples': len(zs_correct),
            'final_forgetting': 0.0,
            'mean_forgetting': 0.0,
            'total_runtime_s': 0.0,
            'avg_step_time_ms': 0.0,
        })
        print(f"  Accuracy: {zs_acc:.2f}%")

        # ── STAD-vMF ────────────────────────────────────────────
        for bs in stad_batch_sizes:
            tag = f'STAD-vMF-bs{bs}'
            print(f"\n--- {tag} ---")
            random.seed(1); torch.manual_seed(1)
            stad_result = run_stad(
                loader=test_loader,
                clip_model=clip_model,
                clip_weights=clip_weights,
                device=device,
                kappa_trans=args.stad_kappa_trans,
                kappa_ems=args.stad_kappa_ems,
                window_size=args.stad_window_size,
                n_em_iters=args.stad_n_em_iters,
                batch_size=bs,
                learn_kappa=args.stad_learn_kappa,
                max_samples=args.max_samples,
                return_details=True,
            )
            all_results.append({
                'method': tag,
                **{k: v for k, v in stad_result.items()
                   if k not in ('cumulative_accuracy', 'forgetting_curve')},
            })
            print(f"  Accuracy: {stad_result['accuracy']:.2f}%")

        # ── Save results ────────────────────────────────────────
        # CSV summary
        csv_path = os.path.join(out_dir, 'comparison.csv')
        fieldnames = ['method', 'accuracy', 'num_samples', 'final_forgetting',
                      'mean_forgetting', 'total_runtime_s', 'avg_step_time_ms']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)

        # JSON with full detail
        json_path = os.path.join(out_dir, 'comparison.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\nResults saved to {out_dir}/")
        print("\n" + "="*60)
        print(f"{'Method':<28} {'Acc':>7} {'Forget':>8} {'Runtime':>9}")
        print("-"*60)
        for r in all_results:
            print(f"{r['method']:<28} {r['accuracy']:>6.2f}% {r.get('mean_forgetting',0):>7.3f} {r.get('total_runtime_s',0):>8.1f}s")
        print("="*60)


if __name__ == '__main__':
    main()
