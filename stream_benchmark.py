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


def parse_str_list(value):
    return [x.strip() for x in value.split(',') if x.strip()]


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
    ssm_modes = parse_str_list(args.ssm_correction_modes)
    run_cache = args.benchmark_mode in ['both', 'cache-only']
    run_ssm = args.benchmark_mode in ['both', 'ssm-only']

    # Consensus gating configs
    consensus_n_views = parse_int_list(args.n_views) if args.n_views.strip() else []
    consensus_top_ks = [int(x) for x in args.top_k.split(',') if x.strip()]
    consensus_thresholds = [float(x) for x in args.consensus_threshold.split(',') if x.strip()]
    run_consensus = len(consensus_n_views) > 0

    for dataset_name in datasets:
        print(f'\n=== Dataset: {dataset_name} ===')
        cfg_base = get_config_file(args.config, dataset_name)

        # Build once to get classnames/template.
        _, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess, shuffle=False)
        clip_weights = clip_classifier(classnames, template, clip_model, device=device)

        for stream_len in stream_lengths:
            print(f'--- Stream length: {stream_len} ---')

            if run_cache:
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

            if run_ssm:
                for ssm_mode in ssm_modes:
                    set_seed(args.seed)
                    test_loader, _, _ = build_test_data_loader(dataset_name, args.data_root, preprocess, shuffle=False)
                    if args.wandb_log:
                        import wandb
                        wandb.init(project=args.wandb_project, name=f"{dataset_name}_ssm_{ssm_mode}_L{stream_len}", reinit=True)

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
                        ssm_kwargs={
                            'correction_mode': ssm_mode,
                            'kalman_q': args.kalman_q,
                            'kalman_r': args.kalman_r,
                            'kalman_q_min': args.kalman_q_min,
                            'kalman_q_max': args.kalman_q_max,
                            'kalman_r_min': args.kalman_r_min,
                            'kalman_r_max': args.kalman_r_max,
                        },
                    )

                    if args.wandb_log:
                        import wandb
                        wandb.finish()

                    method_name = f'tda_ssm_{ssm_mode}'
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

            # Consensus-gated SSM runs
            if run_consensus:
                for n_views in consensus_n_views:
                    for top_k in consensus_top_ks:
                        for cthresh in consensus_thresholds:
                            for ssm_mode in ssm_modes:
                                set_seed(args.seed)
                                test_loader, _, _ = build_test_data_loader(
                                    dataset_name, args.data_root, preprocess,
                                    shuffle=False,
                                    n_views=min(n_views, max(0, args.dynamic_max_views - 1)) if args.dynamic_view_budget else n_views)

                                consensus_kwargs = {
                                    'n_views': min(n_views, max(0, args.dynamic_max_views - 1)) if args.dynamic_view_budget else n_views,
                                    'top_k': top_k,
                                    'consensus_threshold': cthresh,
                                    'dynamic_view_budget': args.dynamic_view_budget,
                                    'dynamic_kwargs': {
                                        'max_views': max(1, args.dynamic_max_views),
                                        'min_views': 2,
                                        'mid_views': max(2, args.dynamic_mid_views),
                                        'single_view_confidence': args.single_view_confidence,
                                        'single_view_margin': args.single_view_margin,
                                        'early_stop_score': args.early_stop_score,
                                        'early_stop_margin': args.early_stop_margin,
                                    },
                                }

                                method_name = f'tda_ssm_{ssm_mode}_cv{consensus_kwargs["n_views"]}_k{top_k}_t{cthresh}'
                                if args.dynamic_view_budget:
                                    method_name += '_dyn'
                                if args.wandb_log:
                                    import wandb
                                    wandb.init(project=args.wandb_project,
                                               name=f"{dataset_name}_{method_name}_L{stream_len}",
                                               reinit=True)

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
                                    ssm_kwargs={
                                        'correction_mode': ssm_mode,
                                        'kalman_q': args.kalman_q,
                                        'kalman_r': args.kalman_r,
                                        'kalman_q_min': args.kalman_q_min,
                                        'kalman_q_max': args.kalman_q_max,
                                        'kalman_r_min': args.kalman_r_min,
                                        'kalman_r_max': args.kalman_r_max,
                                    },
                                    consensus_kwargs=consensus_kwargs,
                                )

                                if args.wandb_log:
                                    import wandb
                                    wandb.finish()

                                peak_mem_mb = result.get('peak_cuda_memory_mb', result.get('peak_mps_memory_mb', result.get('peak_device_memory_mb', 0.0)))
                                row = {
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
                                }
                                if 'consensus_accept_rate' in result:
                                    row['consensus_accept_rate'] = result['consensus_accept_rate']
                                if 'avg_soft_score' in result:
                                    row['avg_soft_score'] = result['avg_soft_score']
                                if 'avg_update_weight' in result:
                                    row['avg_update_weight'] = result['avg_update_weight']
                                if 'avg_views_used' in result:
                                    row['avg_views_used'] = result['avg_views_used']
                                if 'avg_view_savings' in result:
                                    row['avg_view_savings'] = result['avg_view_savings']
                                if 'early_stop_rate' in result:
                                    row['early_stop_rate'] = result['early_stop_rate']
                                summary_rows.append(row)

                                for step in range(0, len(result['cumulative_accuracy']), args.curve_stride):
                                    curve_row = {
                                        'dataset': dataset_name,
                                        'method': method_name,
                                        'stream_length': stream_len,
                                        'step': step + 1,
                                        'cumulative_accuracy': result['cumulative_accuracy'][step],
                                        'forgetting': result['forgetting_curve'][step],
                                    }
                                    if 'consensus_accept_rate_curve' in result:
                                        curve_row['consensus_accept_rate'] = result['consensus_accept_rate_curve'][step]
                                    if 'soft_score_curve' in result:
                                        curve_row['avg_soft_score'] = result['soft_score_curve'][step]
                                    if 'update_weight_curve' in result:
                                        curve_row['avg_update_weight'] = result['update_weight_curve'][step]
                                    if 'avg_views_used_curve' in result:
                                        curve_row['avg_views_used'] = result['avg_views_used_curve'][step]
                                    if 'avg_view_savings_curve' in result:
                                        curve_row['avg_view_savings'] = result['avg_view_savings_curve'][step]
                                    curve_rows.append(curve_row)

    summary_csv = os.path.join(args.output_dir, 'summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        all_keys = []
        for row in summary_rows:
            for k in row:
                if k not in all_keys:
                    all_keys.append(k)
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary_rows)

    curves_csv = os.path.join(args.output_dir, 'curves.csv')
    with open(curves_csv, 'w', newline='') as f:
        curve_keys = []
        for row in curve_rows:
            for k in row:
                if k not in curve_keys:
                    curve_keys.append(k)
        writer = csv.DictWriter(f, fieldnames=curve_keys, extrasaction='ignore')
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
    parser.add_argument('--benchmark-mode', type=str, choices=['both', 'ssm-only', 'cache-only'], default='both', help='Run both families, only SSM variants, or only cache variants.')
    parser.add_argument('--stream-lengths', type=str, default='500,1000,2000,5000', help='Comma-separated stream lengths.')
    parser.add_argument('--cache-sizes', type=str, default='1,2,3,5', help='Comma-separated cache sizes for original TDA.')
    parser.add_argument('--curve-stride', type=int, default=50, help='Store every N points from curves.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for CSV/JSON/plots.')
    parser.add_argument('--ssm-correction-modes', type=str, default='heuristic,kalman-fixed,kalman-adaptive', help='Comma-separated SSM correction variants to benchmark.')
    parser.add_argument('--kalman-q', type=float, default=0.01, help='Fixed Kalman process noise Q for kalman-fixed mode.')
    parser.add_argument('--kalman-r', type=float, default=0.05, help='Fixed Kalman observation noise R for kalman-fixed mode.')
    parser.add_argument('--kalman-q-min', type=float, default=0.005, help='Minimum adaptive Kalman process noise Q.')
    parser.add_argument('--kalman-q-max', type=float, default=0.05, help='Maximum adaptive Kalman process noise Q.')
    parser.add_argument('--kalman-r-min', type=float, default=0.01, help='Minimum adaptive Kalman observation noise R.')
    parser.add_argument('--kalman-r-max', type=float, default=0.1, help='Maximum adaptive Kalman observation noise R.')
    parser.add_argument('--wandb-log', action='store_true', help='Enable wandb logging for each run.')
    parser.add_argument('--wandb-project', type=str, default='TDA-Benchmark', help='Wandb project name.')
    # Consensus gating parameters
    parser.add_argument('--n-views', type=str, default='', help='Comma-separated n_views values for consensus gating (empty=disabled).')
    parser.add_argument('--top-k', type=str, default='5', help='Comma-separated top-K values for consensus masking.')
    parser.add_argument('--consensus-threshold', type=str, default='0.5', help='Comma-separated consensus threshold values (0.0-1.0).')
    parser.add_argument('--dynamic-view-budget', action='store_true', help='Enable dynamic soft-consensus view scheduling with early stopping.')
    parser.add_argument('--dynamic-max-views', type=int, default=8, help='Maximum total views to use in dynamic consensus, including the clean view.')
    parser.add_argument('--dynamic-mid-views', type=int, default=4, help='Intermediate total view budget for medium-confidence samples.')
    parser.add_argument('--single-view-confidence', type=float, default=0.80, help='Confidence score needed to stop after one view in dynamic consensus.')
    parser.add_argument('--single-view-margin', type=float, default=0.35, help='Top-1 minus top-2 probability margin needed to stop after one view.')
    parser.add_argument('--early-stop-score', type=float, default=0.60, help='Soft agreement score needed to stop early after the initial dynamic view budget.')
    parser.add_argument('--early-stop-margin', type=float, default=0.25, help='Probability margin needed to stop early after the initial dynamic view budget.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.output_dir is None:
        date = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join('results', f'stream_benchmark_{date}')
    run_head_to_head(args)
