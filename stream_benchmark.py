import argparse
import copy
import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import torch

import clip
from tda_runner import run_test_tda
from utils import build_test_data_loader, clip_classifier, get_config_file, resolve_device


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_make_plots(summary_rows, curve_rows, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('matplotlib is not available. Skipping plot generation.')
        return

    datasets = sorted(list({row['dataset'] for row in summary_rows}))
    for dataset in datasets:
        ds_rows = [r for r in summary_rows if r['dataset'] == dataset]
        methods = sorted(list({r['method'] for r in ds_rows}))

        plt.figure(figsize=(8, 5))
        for method in methods:
            method_rows = [r for r in ds_rows if r['method'] == method]
            method_rows = sorted(method_rows, key=lambda x: int(x['stream_length']))
            x = [int(r['stream_length']) for r in method_rows]
            y = [float(r['accuracy']) for r in method_rows]
            plt.plot(x, y, marker='o', label=method)

        plt.xlabel('Stream Length')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Accuracy vs Stream Length ({dataset})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'accuracy_vs_stream_length_{dataset}.png'), dpi=180)
        plt.close()

        max_stream = max(int(r['stream_length']) for r in ds_rows)
        curve_ds = [r for r in curve_rows if r['dataset'] == dataset and int(r['stream_length']) == max_stream]
        curve_methods = sorted(list({r['method'] for r in curve_ds}))

        if len(curve_methods) > 0:
            plt.figure(figsize=(8, 5))
            for method in curve_methods:
                method_curve = [r for r in curve_ds if r['method'] == method]
                method_curve = sorted(method_curve, key=lambda x: int(x['step']))
                x = [int(r['step']) for r in method_curve]
                y = [float(r['forgetting']) for r in method_curve]
                plt.plot(x, y, label=method)

            plt.xlabel('Stream Step')
            plt.ylabel('Forgetting (%)')
            plt.title(f'Forgetting Curve at Stream Length={max_stream} ({dataset})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'forgetting_curve_{dataset}_L{max_stream}.png'), dpi=180)
            plt.close()


def run_head_to_head(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device(args.device)
    clip_model, preprocess = clip.load(args.backbone, device=device)
    if device.type == 'mps':
        clip_model.float()
    clip_model.eval()

    summary_rows = []
    curve_rows = []

    datasets = args.datasets.split('/')
    stream_lengths = parse_int_list(args.stream_lengths)
    cache_sizes = parse_int_list(args.cache_sizes)

    for dataset_name in datasets:
        print(f'\n=== Dataset: {dataset_name} ===')
        cfg_base = get_config_file(args.config, dataset_name)

        # Build once to get classnames/template.
        _, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess, shuffle=False)
        clip_weights = clip_classifier(classnames, template, clip_model, device=device)

        for stream_len in stream_lengths:
            print(f'--- Stream length: {stream_len} ---')

            for cache_size in cache_sizes:
                cfg = copy.deepcopy(cfg_base)
                if cfg['positive']['enabled']:
                    cfg['positive']['shot_capacity'] = cache_size
                if cfg['negative']['enabled']:
                    cfg['negative']['shot_capacity'] = cache_size

                set_seed(args.seed)
                test_loader, _, _ = build_test_data_loader(dataset_name, args.data_root, preprocess, shuffle=False)
                if args.wandb_log:
                    import wandb
                    wandb.init(project=args.wandb_project, name=f"{dataset_name}_cache_k{cache_size}_L{stream_len}", reinit=True)

                result = run_test_tda(
                    cfg['positive'],
                    cfg['negative'],
                    test_loader,
                    clip_model,
                    clip_weights,
                    memory_type='cache',
                    max_samples=stream_len,
                    enable_wandb=args.wandb_log,
                    log_interval=max(1, stream_len // 5),
                    return_details=True,
                )

                if args.wandb_log:
                    import wandb
                    wandb.finish()

                method_name = f'tda_cache_k{cache_size}'
                peak_mem_mb = result.get('peak_cuda_memory_mb', result.get('peak_mps_memory_mb', result.get('peak_device_memory_mb', 0.0)))
                summary_rows.append({
                    'dataset': dataset_name,
                    'method': method_name,
                    'stream_length': stream_len,
                    'cache_size': cache_size,
                    'accuracy': result['accuracy'],
                    'final_forgetting': result['final_forgetting'],
                    'mean_forgetting': result['mean_forgetting'],
                    'avg_step_time_ms': result['avg_step_time_ms'],
                    'total_runtime_s': result['total_runtime_s'],
                    'peak_device_memory_mb': peak_mem_mb,
                    'adapter_memory_mb': result['adapter_memory_mb'],
                    'num_samples': result['num_samples'],
                })

                for step in range(0, len(result['cumulative_accuracy']), args.curve_stride):
                    curve_rows.append({
                        'dataset': dataset_name,
                        'method': method_name,
                        'stream_length': stream_len,
                        'step': step + 1,
                        'cumulative_accuracy': result['cumulative_accuracy'][step],
                        'forgetting': result['forgetting_curve'][step],
                    })

            set_seed(args.seed)
            test_loader, _, _ = build_test_data_loader(dataset_name, args.data_root, preprocess, shuffle=False)
            if args.wandb_log:
                import wandb
                wandb.init(project=args.wandb_project, name=f"{dataset_name}_ssm_L{stream_len}", reinit=True)

            result = run_test_tda(
                cfg_base['positive'],
                cfg_base['negative'],
                test_loader,
                clip_model,
                clip_weights,
                memory_type='ssm',
                max_samples=stream_len,
                enable_wandb=args.wandb_log,
                log_interval=max(1, stream_len // 5),
                return_details=True,
            )

            if args.wandb_log:
                import wandb
                wandb.finish()

            method_name = 'tda_ssm'
            peak_mem_mb = result.get('peak_cuda_memory_mb', result.get('peak_mps_memory_mb', result.get('peak_device_memory_mb', 0.0)))
            summary_rows.append({
                'dataset': dataset_name,
                'method': method_name,
                'stream_length': stream_len,
                'cache_size': -1,
                'accuracy': result['accuracy'],
                'final_forgetting': result['final_forgetting'],
                'mean_forgetting': result['mean_forgetting'],
                'avg_step_time_ms': result['avg_step_time_ms'],
                'total_runtime_s': result['total_runtime_s'],
                'peak_device_memory_mb': peak_mem_mb,
                'adapter_memory_mb': result['adapter_memory_mb'],
                'num_samples': result['num_samples'],
            })

            for step in range(0, len(result['cumulative_accuracy']), args.curve_stride):
                curve_rows.append({
                    'dataset': dataset_name,
                    'method': method_name,
                    'stream_length': stream_len,
                    'step': step + 1,
                    'cumulative_accuracy': result['cumulative_accuracy'][step],
                    'forgetting': result['forgetting_curve'][step],
                })

    summary_csv = os.path.join(args.output_dir, 'summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    curves_csv = os.path.join(args.output_dir, 'curves.csv')
    with open(curves_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)

    report_json = os.path.join(args.output_dir, 'report.json')
    with open(report_json, 'w') as f:
        json.dump({'summary': summary_rows, 'curves': curve_rows}, f)

    maybe_make_plots(summary_rows, curve_rows, args.output_dir)

    print(f'\nSaved summary to: {summary_csv}')
    print(f'Saved curves to: {curves_csv}')
    print(f'Saved report to: {report_json}')


def get_args():
    parser = argparse.ArgumentParser(description='Head-to-head streaming benchmark: TDA cache vs SSM memory.')
    parser.add_argument('--config', type=str, required=True, help='Path to config directory (e.g. configs).')
    parser.add_argument('--datasets', type=str, required=True, help='Datasets, separated by / (e.g. I/A/V/R/S).')
    parser.add_argument('--data-root', type=str, default='./dataset/', help='Dataset root path.')
    parser.add_argument('--backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP backbone.')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'mps', 'cpu'], default='auto', help='Execution device. Default: auto.')
    parser.add_argument('--stream-lengths', type=str, default='500,1000,2000,5000', help='Comma-separated stream lengths.')
    parser.add_argument('--cache-sizes', type=str, default='1,2,3,5', help='Comma-separated cache sizes for original TDA.')
    parser.add_argument('--curve-stride', type=int, default=50, help='Store every N points from curves.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for CSV/JSON/plots.')
    parser.add_argument('--wandb-log', action='store_true', help='Enable wandb logging for each run.')
    parser.add_argument('--wandb-project', type=str, default='TDA-Benchmark', help='Wandb project name.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.output_dir is None:
        date = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join('results', f'stream_benchmark_{date}')
    run_head_to_head(args)
