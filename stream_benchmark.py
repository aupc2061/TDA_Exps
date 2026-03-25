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
    anchor_modes = parse_str_list(args.anchor_modes)
    anchor_reservoir_sizes = parse_int_list(args.anchor_reservoir_sizes)
    anchor_alphas = [float(x) for x in args.anchor_alphas.split(',') if x.strip()]
    anchor_betas = [float(x) for x in args.anchor_betas.split(',') if x.strip()]
    anchor_entropy_thresholds = [float(x) for x in args.anchor_entropy_thresholds.split(',') if x.strip()]
    mamba3_modes = parse_str_list(args.mamba3_modes)
    mamba3_phase_strengths = [float(x) for x in args.mamba3_phase_strengths.split(',') if x.strip()]
    mamba3_num_slots = parse_int_list(args.mamba3_num_slots)
    mamba3_new_slot_thresholds = [float(x) for x in args.mamba3_new_slot_thresholds.split(',') if x.strip()]

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

            # if run_ssm:
            #     for ssm_mode in ssm_modes:
                    # set_seed(args.seed)
                    # test_loader, _, _ = build_test_data_loader(dataset_name, args.data_root, preprocess, shuffle=False)
                    # if args.wandb_log:
                    #     import wandb
                    #     wandb.init(project=args.wandb_project, name=f"{dataset_name}_ssm_{ssm_mode}_L{stream_len}", reinit=True)

                    # result = run_test_tda(
                    #     cfg_base['positive'],
                    #     cfg_base['negative'],
                    #     test_loader,
                    #     clip_model,
                    #     clip_weights,
                    #     memory_type='ssm',
                    #     max_samples=stream_len,
                    #     enable_wandb=args.wandb_log,
                    #     log_interval=max(1, stream_len // 5),
                    #     return_details=True,
                    #     ssm_kwargs={
                    #         'correction_mode': ssm_mode,
                    #         'kalman_q': args.kalman_q,
                    #         'kalman_r': args.kalman_r,
                    #         'kalman_q_min': args.kalman_q_min,
                    #         'kalman_q_max': args.kalman_q_max,
                    #         'kalman_r_min': args.kalman_r_min,
                    #         'kalman_r_max': args.kalman_r_max,
                    #     },
                    # )

                    # if args.wandb_log:
                    #     import wandb
                    #     wandb.finish()

                    # method_name = f'tda_ssm_{ssm_mode}'
                    # peak_mem_mb = result.get('peak_cuda_memory_mb', result.get('peak_mps_memory_mb', result.get('peak_device_memory_mb', 0.0)))
                    # summary_rows.append({
                    #     'dataset': dataset_name,
                    #     'method': method_name,
                    #     'stream_length': stream_len,
                    #     'cache_size': -1,
                    #     'accuracy': result['accuracy'],
                    #     'final_forgetting': result['final_forgetting'],
                    #     'mean_forgetting': result['mean_forgetting'],
                    #     'avg_step_time_ms': result['avg_step_time_ms'],
                    #     'total_runtime_s': result['total_runtime_s'],
                    #     'peak_device_memory_mb': peak_mem_mb,
                    #     'adapter_memory_mb': result['adapter_memory_mb'],
                    #     'num_samples': result['num_samples'],
                    # })

                    # for step in range(0, len(result['cumulative_accuracy']), args.curve_stride):
                    #     curve_rows.append({
                    #         'dataset': dataset_name,
                    #         'method': method_name,
                    #         'stream_length': stream_len,
                    #         'step': step + 1,
                    #         'cumulative_accuracy': result['cumulative_accuracy'][step],
                    #         'forgetting': result['forgetting_curve'][step],
                    #     })

            #         if 'on' in anchor_modes:
            #             for reservoir_size in anchor_reservoir_sizes:
            #                 for anchor_alpha in anchor_alphas:
            #                     for anchor_beta in anchor_betas:
            #                         for anchor_entropy_threshold in anchor_entropy_thresholds:
            #                             set_seed(args.seed)
            #                             test_loader, _, _ = build_test_data_loader(dataset_name, args.data_root, preprocess, shuffle=False)
            #                             if args.wandb_log:
            #                                 import wandb
            #                                 wandb.init(
            #                                     project=args.wandb_project,
            #                                     name=(
            #                                         f"{dataset_name}_ssm_{ssm_mode}_anchor"
            #                                         f"_r{reservoir_size}_a{anchor_alpha}_b{anchor_beta}_e{anchor_entropy_threshold}_L{stream_len}"
            #                                     ),
            #                                     reinit=True,
            #                                 )

            #                             anchor_result = run_test_tda(
            #                                 cfg_base['positive'],
            #                                 cfg_base['negative'],
            #                                 test_loader,
            #                                 clip_model,
            #                                 clip_weights,
            #                                 memory_type='ssm',
            #                                 max_samples=stream_len,
            #                                 enable_wandb=args.wandb_log,
            #                                 log_interval=max(1, stream_len // 5),
            #                                 return_details=True,
            #                                 ssm_kwargs={
            #                                     'correction_mode': ssm_mode,
            #                                     'kalman_q': args.kalman_q,
            #                                     'kalman_r': args.kalman_r,
            #                                     'kalman_q_min': args.kalman_q_min,
            #                                     'kalman_q_max': args.kalman_q_max,
            #                                     'kalman_r_min': args.kalman_r_min,
            #                                     'kalman_r_max': args.kalman_r_max,
            #                                 },
            #                                 anchor_kwargs={
            #                                     'enabled': True,
            #                                     'capacity': reservoir_size,
            #                                     'entropy_threshold': anchor_entropy_threshold,
            #                                     'alpha': anchor_alpha,
            #                                     'beta': anchor_beta,
            #                                 },
            #                             )

            #                             if args.wandb_log:
            #                                 import wandb
            #                                 wandb.finish()

            #                             anchor_method_name = (
            #                                 f'tda_ssm_{ssm_mode}_anchor'
            #                                 f'_r{reservoir_size}_a{anchor_alpha}_b{anchor_beta}_e{anchor_entropy_threshold}'
            #                             )
            #                             peak_mem_mb = anchor_result.get('peak_cuda_memory_mb', anchor_result.get('peak_mps_memory_mb', anchor_result.get('peak_device_memory_mb', 0.0)))
            #                             row = {
            #                                 'dataset': dataset_name,
            #                                 'method': anchor_method_name,
            #                                 'stream_length': stream_len,
            #                                 'cache_size': -1,
            #                                 'accuracy': anchor_result['accuracy'],
            #                                 'final_forgetting': anchor_result['final_forgetting'],
            #                                 'mean_forgetting': anchor_result['mean_forgetting'],
            #                                 'avg_step_time_ms': anchor_result['avg_step_time_ms'],
            #                                 'total_runtime_s': anchor_result['total_runtime_s'],
            #                                 'peak_device_memory_mb': peak_mem_mb,
            #                                 'adapter_memory_mb': anchor_result['adapter_memory_mb'],
            #                                 'num_samples': anchor_result['num_samples'],
            #                                 'anchor_reservoir_size': reservoir_size,
            #                                 'anchor_alpha': anchor_alpha,
            #                                 'anchor_beta': anchor_beta,
            #                                 'anchor_entropy_threshold': anchor_entropy_threshold,
            #                             }
            #                             for key in [
            #                                 'anchor_update_accept_count',
            #                                 'anchor_update_attempt_count',
            #                                 'anchor_update_accept_rate',
            #                                 'anchor_fill_ratio',
            #                                 'avg_anchor_fill_ratio',
            #                                 'avg_anchor_correction_active_rate',
            #                                 'avg_anchor_correction_norm',
            #                                 'anchor_invalid_update_rejections',
            #                                 'anchor_invalid_query_rejections',
            #                                 'anchor_invalid_anchor_rejections',
            #                                 'anchor_correction_skip_count',
            #                             ]:
            #                                 if key in anchor_result:
            #                                     row[key] = anchor_result[key]
            #                             summary_rows.append(row)

            #                             for step in range(0, len(anchor_result['cumulative_accuracy']), args.curve_stride):
            #                                 curve_row = {
            #                                     'dataset': dataset_name,
            #                                     'method': anchor_method_name,
            #                                     'stream_length': stream_len,
            #                                     'step': step + 1,
            #                                     'cumulative_accuracy': anchor_result['cumulative_accuracy'][step],
            #                                     'forgetting': anchor_result['forgetting_curve'][step],
            #                                 }
            #                                 if 'anchor_update_accept_rate_curve' in anchor_result:
            #                                     curve_row['anchor_update_accept_rate'] = anchor_result['anchor_update_accept_rate_curve'][step]
            #                                 if 'anchor_fill_ratio_curve' in anchor_result:
            #                                     curve_row['anchor_fill_ratio'] = anchor_result['anchor_fill_ratio_curve'][step]
            #                                 if 'anchor_correction_active_rate_curve' in anchor_result:
            #                                     curve_row['anchor_correction_active_rate'] = anchor_result['anchor_correction_active_rate_curve'][step]
            #                                 if 'anchor_correction_norm_curve' in anchor_result:
            #                                     curve_row['anchor_correction_norm'] = anchor_result['anchor_correction_norm_curve'][step]
            #                                 curve_rows.append(curve_row)

            # if len(mamba3_modes) > 0:
            #     for mamba3_mode in mamba3_modes:
            #         phase_strength_values = mamba3_phase_strengths if mamba3_mode == 'mamba3-complex' else [0.0]
            #         slot_values = mamba3_num_slots if mamba3_mode == 'mamba3-multislot' else [1]
            #         for phase_strength in phase_strength_values:
            #             for num_slots in slot_values:
            #                 for slot_threshold in mamba3_new_slot_thresholds:
            #                     anchor_cfgs = [None]
            #                     if 'on' in anchor_modes:
            #                         anchor_cfgs.extend([
            #                             {
            #                                 'capacity': reservoir_size,
            #                                 'entropy_threshold': anchor_entropy_threshold,
            #                                 'alpha': anchor_alpha,
            #                                 'beta': anchor_beta,
            #                             }
            #                             for reservoir_size in anchor_reservoir_sizes
            #                             for anchor_alpha in anchor_alphas
            #                             for anchor_beta in anchor_betas
            #                             for anchor_entropy_threshold in anchor_entropy_thresholds
            #                         ])

            #                     for anchor_cfg in anchor_cfgs:
            #                         set_seed(args.seed)
            #                         test_loader, _, _ = build_test_data_loader(dataset_name, args.data_root, preprocess, shuffle=False)
            #                         if args.wandb_log:
            #                             import wandb
            #                             run_name = (
            #                                 f"{dataset_name}_{mamba3_mode}"
            #                                 f"_ps{phase_strength}_slots{num_slots}_thr{slot_threshold}"
            #                             )
            #                             if anchor_cfg is not None:
            #                                 run_name += (
            #                                     f"_anchor_r{anchor_cfg['capacity']}"
            #                                     f"_a{anchor_cfg['alpha']}"
            #                                     f"_b{anchor_cfg['beta']}"
            #                                     f"_e{anchor_cfg['entropy_threshold']}"
            #                                 )
            #                             wandb.init(project=args.wandb_project, name=f"{run_name}_L{stream_len}", reinit=True)

            #                         result = run_test_tda(
            #                             cfg_base['positive'],
            #                             cfg_base['negative'],
            #                             test_loader,
            #                             clip_model,
            #                             clip_weights,
            #                             memory_type='ssm-mamba3',
            #                             max_samples=stream_len,
            #                             enable_wandb=args.wandb_log,
            #                             log_interval=max(1, stream_len // 5),
            #                             return_details=True,
            #                             ssm_kwargs={
            #                                 'correction_mode': args.ssm_correction_modes.split(',')[0].strip() if args.ssm_correction_modes else 'heuristic',
            #                                 'kalman_q': args.kalman_q,
            #                                 'kalman_r': args.kalman_r,
            #                                 'kalman_q_min': args.kalman_q_min,
            #                                 'kalman_q_max': args.kalman_q_max,
            #                                 'kalman_r_min': args.kalman_r_min,
            #                                 'kalman_r_max': args.kalman_r_max,
            #                                 'mamba3_mode': mamba3_mode,
            #                                 'mamba3_min_blend': args.mamba3_min_blend,
            #                                 'mamba3_max_blend': args.mamba3_max_blend,
            #                                 'mamba3_phase_strength': phase_strength,
            #                                 'mamba3_num_slots': num_slots,
            #                                 'mamba3_new_slot_threshold': slot_threshold,
            #                             },
            #                             anchor_kwargs={
            #                                 'enabled': True,
            #                                 'capacity': anchor_cfg['capacity'],
            #                                 'entropy_threshold': anchor_cfg['entropy_threshold'],
            #                                 'alpha': anchor_cfg['alpha'],
            #                                 'beta': anchor_cfg['beta'],
            #                             } if anchor_cfg is not None else None,
            #                         )

            #                         if args.wandb_log:
            #                             import wandb
            #                             wandb.finish()

            #                         method_name = (
            #                             f'{mamba3_mode}'
            #                             f'_ps{phase_strength}_slots{num_slots}_thr{slot_threshold}'
            #                         )
            #                         if anchor_cfg is not None:
            #                             method_name += (
            #                                 f"_anchor_r{anchor_cfg['capacity']}"
            #                                 f"_a{anchor_cfg['alpha']}"
            #                                 f"_b{anchor_cfg['beta']}"
            #                                 f"_e{anchor_cfg['entropy_threshold']}"
            #                             )
            #                         peak_mem_mb = result.get('peak_cuda_memory_mb', result.get('peak_mps_memory_mb', result.get('peak_device_memory_mb', 0.0)))
            #                         row = {
            #                             'dataset': dataset_name,
            #                             'method': method_name,
            #                             'stream_length': stream_len,
            #                             'cache_size': -1,
            #                             'accuracy': result['accuracy'],
            #                             'final_forgetting': result['final_forgetting'],
            #                             'mean_forgetting': result['mean_forgetting'],
            #                             'avg_step_time_ms': result['avg_step_time_ms'],
            #                             'total_runtime_s': result['total_runtime_s'],
            #                             'peak_device_memory_mb': peak_mem_mb,
            #                             'adapter_memory_mb': result['adapter_memory_mb'],
            #                             'num_samples': result['num_samples'],
            #                             'mamba3_mode': mamba3_mode,
            #                             'mamba3_phase_strength': phase_strength,
            #                             'mamba3_num_slots': num_slots,
            #                             'mamba3_new_slot_threshold': slot_threshold,
            #                         }
            #                         if anchor_cfg is not None:
            #                             row['anchor_reservoir_size'] = anchor_cfg['capacity']
            #                             row['anchor_alpha'] = anchor_cfg['alpha']
            #                             row['anchor_beta'] = anchor_cfg['beta']
            #                             row['anchor_entropy_threshold'] = anchor_cfg['entropy_threshold']
            #                         for key in [
            #                             'mamba3_update_accept_count',
            #                             'mamba3_update_attempt_count',
            #                             'mamba3_update_accept_rate',
            #                             'mamba3_active_slot_ratio',
            #                             'mamba3_avg_slot_utilization',
            #                             'mamba3_avg_phase_magnitude',
            #                             'mamba3_avg_alpha',
            #                             'mamba3_avg_beta',
            #                             'mamba3_avg_gamma',
            #                             'anchor_update_accept_count',
            #                             'anchor_update_attempt_count',
            #                             'anchor_update_accept_rate',
            #                             'anchor_fill_ratio',
            #                             'avg_anchor_fill_ratio',
            #                             'avg_anchor_correction_active_rate',
            #                             'avg_anchor_correction_norm',
            #                         ]:
            #                             if key in result:
            #                                 row[key] = result[key]
            #                         summary_rows.append(row)

            #                         for step in range(0, len(result['cumulative_accuracy']), args.curve_stride):
            #                             curve_row = {
            #                                 'dataset': dataset_name,
            #                                 'method': method_name,
            #                                 'stream_length': stream_len,
            #                                 'step': step + 1,
            #                                 'cumulative_accuracy': result['cumulative_accuracy'][step],
            #                                 'forgetting': result['forgetting_curve'][step],
            #                             }
            #                             if 'anchor_update_accept_rate_curve' in result:
            #                                 curve_row['anchor_update_accept_rate'] = result['anchor_update_accept_rate_curve'][step]
            #                             if 'anchor_fill_ratio_curve' in result:
            #                                 curve_row['anchor_fill_ratio'] = result['anchor_fill_ratio_curve'][step]
            #                             if 'anchor_correction_active_rate_curve' in result:
            #                                 curve_row['anchor_correction_active_rate'] = result['anchor_correction_active_rate_curve'][step]
            #                             if 'anchor_correction_norm_curve' in result:
            #                                 curve_row['anchor_correction_norm'] = result['anchor_correction_norm_curve'][step]
            #                             curve_rows.append(curve_row)

            if run_ssm:
                for ssm_mode in ssm_modes:
                    anchor_cfgs = []
                    if 'off' in anchor_modes:
                        anchor_cfgs.append(None)
                    if 'on' in anchor_modes:
                        anchor_cfgs.extend([
                            {
                                'capacity': reservoir_size,
                                'entropy_threshold': anchor_entropy_threshold,
                                'alpha': anchor_alpha,
                                'beta': anchor_beta,
                            }
                            for reservoir_size in anchor_reservoir_sizes
                            for anchor_alpha in anchor_alphas
                            for anchor_beta in anchor_betas
                            for anchor_entropy_threshold in anchor_entropy_thresholds
                        ])

                    for anchor_cfg in anchor_cfgs:
                        set_seed(args.seed)
                        test_loader, _, _ = build_test_data_loader(dataset_name, args.data_root, preprocess, shuffle=False)

                        method_name = f'tda_ssm_{ssm_mode}'
                        if anchor_cfg is not None:
                            method_name += (
                                f'_anchor_r{anchor_cfg["capacity"]}'
                                f'_a{anchor_cfg["alpha"]}'
                                f'_b{anchor_cfg["beta"]}'
                                f'_e{anchor_cfg["entropy_threshold"]}'
                            )
                        if args.enable_token_condensation:
                            method_name += (
                                f'_tc_kr{args.token_keep_ratio}'
                                f'_et{args.token_condense_entropy_threshold}'
                                f'_mt{args.token_condense_margin_threshold}'
                            )

                        if args.wandb_log:
                            import wandb
                            wandb.init(project=args.wandb_project, name=f"{dataset_name}_{method_name}_L{stream_len}", reinit=True)

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
                            anchor_kwargs={
                                'enabled': True,
                                'capacity': anchor_cfg['capacity'],
                                'entropy_threshold': anchor_cfg['entropy_threshold'],
                                'alpha': anchor_cfg['alpha'],
                                'beta': anchor_cfg['beta'],
                            } if anchor_cfg is not None else None,
                            token_condense_kwargs={
                                'enabled': args.enable_token_condensation,
                                'backbone_scope': args.token_condense_backbone,
                                'entropy_threshold': args.token_condense_entropy_threshold,
                                'margin_threshold': args.token_condense_margin_threshold,
                                'consensus_threshold': args.token_condense_consensus_threshold,
                                'layers': [int(x.strip()) for x in args.token_condense_layers.split(',') if x.strip()],
                                'keep_ratio': args.token_keep_ratio,
                                'merge_ratio': args.token_merge_ratio,
                                'max_samples_debug': args.token_condense_max_samples_debug,
                                'min_keep_tokens': args.token_condense_min_keep_tokens,
                            } if args.enable_token_condensation else None,
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
                        if anchor_cfg is not None:
                            row['anchor_reservoir_size'] = anchor_cfg['capacity']
                            row['anchor_alpha'] = anchor_cfg['alpha']
                            row['anchor_beta'] = anchor_cfg['beta']
                            row['anchor_entropy_threshold'] = anchor_cfg['entropy_threshold']
                        if args.enable_token_condensation:
                            row['token_keep_ratio'] = args.token_keep_ratio
                            row['token_condense_entropy_threshold'] = args.token_condense_entropy_threshold
                            row['token_condense_margin_threshold'] = args.token_condense_margin_threshold
                            row['token_condense_layers'] = args.token_condense_layers
                        for key in [
                            'anchor_update_accept_count',
                            'anchor_update_attempt_count',
                            'anchor_update_accept_rate',
                            'anchor_fill_ratio',
                            'avg_anchor_fill_ratio',
                            'avg_anchor_correction_active_rate',
                            'avg_anchor_correction_norm',
                            'anchor_invalid_update_rejections',
                            'anchor_invalid_query_rejections',
                            'anchor_invalid_anchor_rejections',
                            'anchor_correction_skip_count',
                            'token_condense_attempt_count',
                            'token_condense_apply_count',
                            'token_condense_attempt_rate',
                            'token_condense_apply_rate',
                            'avg_token_condense_keep_ratio',
                            'avg_token_condense_kept_tokens',
                            'avg_token_condense_layers',
                            'avg_token_condense_fallbacks',
                        ]:
                            if key in result:
                                row[key] = result[key]
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
                            if 'anchor_update_accept_rate_curve' in result:
                                curve_row['anchor_update_accept_rate'] = result['anchor_update_accept_rate_curve'][step]
                            if 'anchor_fill_ratio_curve' in result:
                                curve_row['anchor_fill_ratio'] = result['anchor_fill_ratio_curve'][step]
                            if 'anchor_correction_active_rate_curve' in result:
                                curve_row['anchor_correction_active_rate'] = result['anchor_correction_active_rate_curve'][step]
                            if 'anchor_correction_norm_curve' in result:
                                curve_row['anchor_correction_norm'] = result['anchor_correction_norm_curve'][step]
                            if 'token_condense_attempt_rate_curve' in result:
                                curve_row['token_condense_attempt_rate'] = result['token_condense_attempt_rate_curve'][step]
                            if 'token_condense_apply_rate_curve' in result:
                                curve_row['token_condense_apply_rate'] = result['token_condense_apply_rate_curve'][step]
                            curve_rows.append(curve_row)

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
                                if args.consensus_enable_anchor_reservoir:
                                    method_name += (
                                        f'_anchor_r{args.consensus_anchor_reservoir_size}'
                                        f'_a{args.consensus_anchor_alpha}'
                                        f'_b{args.consensus_anchor_beta}'
                                        f'_e{args.consensus_anchor_entropy_threshold}'
                                    )
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
                                    anchor_kwargs={
                                        'enabled': True,
                                        'capacity': args.consensus_anchor_reservoir_size,
                                        'entropy_threshold': args.consensus_anchor_entropy_threshold,
                                        'alpha': args.consensus_anchor_alpha,
                                        'beta': args.consensus_anchor_beta,
                                    } if args.consensus_enable_anchor_reservoir else None,
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
                                if args.consensus_enable_anchor_reservoir:
                                    row['anchor_reservoir_size'] = args.consensus_anchor_reservoir_size
                                    row['anchor_alpha'] = args.consensus_anchor_alpha
                                    row['anchor_beta'] = args.consensus_anchor_beta
                                    row['anchor_entropy_threshold'] = args.consensus_anchor_entropy_threshold
                                for key in [
                                    'anchor_update_accept_count',
                                    'anchor_update_attempt_count',
                                    'anchor_update_accept_rate',
                                    'anchor_fill_ratio',
                                    'avg_anchor_fill_ratio',
                                    'avg_anchor_correction_active_rate',
                                    'avg_anchor_correction_norm',
                                    'anchor_invalid_update_rejections',
                                    'anchor_invalid_query_rejections',
                                    'anchor_invalid_anchor_rejections',
                                    'anchor_correction_skip_count',
                                ]:
                                    if key in result:
                                        row[key] = result[key]
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
                                    if 'anchor_update_accept_rate_curve' in result:
                                        curve_row['anchor_update_accept_rate'] = result['anchor_update_accept_rate_curve'][step]
                                    if 'anchor_fill_ratio_curve' in result:
                                        curve_row['anchor_fill_ratio'] = result['anchor_fill_ratio_curve'][step]
                                    if 'anchor_correction_active_rate_curve' in result:
                                        curve_row['anchor_correction_active_rate'] = result['anchor_correction_active_rate_curve'][step]
                                    if 'anchor_correction_norm_curve' in result:
                                        curve_row['anchor_correction_norm'] = result['anchor_correction_norm_curve'][step]
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
    parser.add_argument('--anchor-modes', type=str, default='off,on', help='Comma-separated anchor benchmark modes. Use off,on to include anchor-enabled runs.')
    parser.add_argument('--anchor-reservoir-sizes', type=str, default='4', help='Comma-separated anchor reservoir sizes.')
    parser.add_argument('--anchor-alphas', type=str, default='0.6', help='Comma-separated anchor correction strengths.')
    parser.add_argument('--anchor-betas', type=str, default='1.5', help='Comma-separated anchor layer weighting temperatures.')
    parser.add_argument('--anchor-entropy-thresholds', type=str, default='0.25', help='Comma-separated anchor entropy thresholds.')
    parser.add_argument('--mamba3-modes', type=str, default='', help='Comma-separated Mamba-3-inspired modes to benchmark (empty=disabled).')
    parser.add_argument('--mamba3-min-blend', type=float, default=0.02, help='Minimum selective blend factor for Mamba-3-inspired runs.')
    parser.add_argument('--mamba3-max-blend', type=float, default=0.35, help='Maximum selective blend factor for Mamba-3-inspired runs.')
    parser.add_argument('--mamba3-phase-strengths', type=str, default='0.15', help='Comma-separated phase strengths for Mamba-3-inspired runs.')
    parser.add_argument('--mamba3-num-slots', type=str, default='4', help='Comma-separated slot counts for Mamba-3-inspired runs.')
    parser.add_argument('--mamba3-new-slot-thresholds', type=str, default='0.25', help='Comma-separated slot allocation thresholds for Mamba-3-inspired runs.')
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
    parser.add_argument('--consensus-enable-anchor-reservoir', action='store_true', help='Enable anchor reservoir for consensus-gated runs.')
    parser.add_argument('--consensus-anchor-reservoir-size', type=int, default=4, help='Anchor reservoir size for consensus-gated runs.')
    parser.add_argument('--consensus-anchor-entropy-threshold', type=float, default=0.25, help='Anchor entropy threshold for consensus-gated runs.')
    parser.add_argument('--consensus-anchor-alpha', type=float, default=0.6, help='Anchor correction strength for consensus-gated runs.')
    parser.add_argument('--consensus-anchor-beta', type=float, default=1.5, help='Anchor layer weighting temperature for consensus-gated runs.')
    parser.add_argument('--enable-token-condensation', action='store_true', help='Enable uncertainty-triggered token condensation for single-view SSM runs.')
    parser.add_argument('--token-condense-backbone', type=str, choices=['vit-only'], default='vit-only', help='Backbone scope for token condensation.')
    parser.add_argument('--token-condense-entropy-threshold', type=float, default=0.25, help='Normalized entropy threshold for token condensation triggering.')
    parser.add_argument('--token-condense-margin-threshold', type=float, default=0.20, help='Top-1/top-2 margin threshold for token condensation triggering.')
    parser.add_argument('--token-condense-consensus-threshold', type=float, default=0.50, help='Consensus threshold placeholder for token condensation.')
    parser.add_argument('--token-condense-layers', type=str, default='8,9,10,11', help='Comma-separated ViT layers where pruning can be applied.')
    parser.add_argument('--token-keep-ratio', type=float, default=0.6, help='Fraction of patch tokens to keep when condensation is applied.')
    parser.add_argument('--token-merge-ratio', type=float, default=0.0, help='Reserved for future token merging support.')
    parser.add_argument('--token-condense-max-samples-debug', type=int, default=None, help='Optional cap on condensed samples for debugging.')
    parser.add_argument('--token-condense-min-keep-tokens', type=int, default=16, help='Minimum number of patch tokens to preserve during pruning.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.output_dir is None:
        date = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join('results', f'stream_benchmark_{date}')
    run_head_to_head(args)
