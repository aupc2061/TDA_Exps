import random
import argparse
from tqdm import tqdm
from datetime import datetime
import time
import importlib

import torch

import clip
from utils import *
from memory import AnchorReservoir, CacheMemory, Mamba3Memory, SSMemory

wandb = None


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')
    parser.add_argument('--device', dest='device', type=str, choices=['auto', 'cuda', 'mps', 'cpu'], default='auto', help='Execution device. Default: auto')
    parser.add_argument('--memory-type', dest='memory_type', type=str, choices=['cache', 'ssm', 'ssm-mamba3'], default='cache', help='Adapter memory backend: original finite cache, current SSM, or Mamba-3-inspired SSM. Default is cache.')
    parser.add_argument('--ssm-correction-mode', dest='ssm_correction_mode', type=str, choices=['heuristic', 'kalman-fixed', 'kalman-adaptive', 'kalman-decoupled', 'vmf-fixed', 'vmf-adaptive', 'vmf-online'], default='heuristic', help='SSM state correction: heuristic, vMF-fixed, vMF-adaptive, vMF-online (Kalman names remain as aliases). Default: heuristic.')
    parser.add_argument('--kalman-q', dest='kalman_q', type=float, default=0.01, help='Fixed Kalman process noise (Q) for --ssm-correction-mode kalman-fixed.')
    parser.add_argument('--kalman-r', dest='kalman_r', type=float, default=0.05, help='Fixed Kalman observation noise (R) for --ssm-correction-mode kalman-fixed.')
    parser.add_argument('--kalman-q-min', dest='kalman_q_min', type=float, default=0.005, help='Minimum adaptive Kalman process noise Q (low novelty).')
    parser.add_argument('--kalman-q-max', dest='kalman_q_max', type=float, default=0.05, help='Maximum adaptive Kalman process noise Q (high novelty).')
    parser.add_argument('--kalman-r-min', dest='kalman_r_min', type=float, default=0.01, help='Minimum adaptive Kalman observation noise R (high confidence).')
    parser.add_argument('--kalman-r-max', dest='kalman_r_max', type=float, default=0.1, help='Maximum adaptive Kalman observation noise R (low confidence).')
    parser.add_argument('--max-samples', dest='max_samples', type=int, default=None, help='Optional maximum stream length per dataset.')
    parser.add_argument('--n-views', dest='n_views', type=int, default=0, help='Number of augmented views for consensus gating (0=disabled).')
    parser.add_argument('--top-k', dest='top_k', type=int, default=5, help='Top-K classes to keep per view in consensus gating.')
    parser.add_argument('--consensus-threshold', dest='consensus_threshold', type=float, default=0.5, help='Fraction of views that must agree for consensus (0.0-1.0).')
    parser.add_argument('--dynamic-view-budget', dest='dynamic_view_budget', action='store_true', help='Enable dynamic soft-consensus view scheduling with early stopping.')
    parser.add_argument('--dynamic-max-views', dest='dynamic_max_views', type=int, default=8, help='Maximum total views to use in dynamic consensus, including the clean view.')
    parser.add_argument('--dynamic-mid-views', dest='dynamic_mid_views', type=int, default=4, help='Intermediate total view budget for medium-confidence samples.')
    parser.add_argument('--single-view-confidence', dest='single_view_confidence', type=float, default=0.80, help='Confidence score needed to stop after one view in dynamic consensus.')
    parser.add_argument('--single-view-margin', dest='single_view_margin', type=float, default=0.35, help='Top-1 minus top-2 probability margin needed to stop after one view.')
    parser.add_argument('--early-stop-score', dest='early_stop_score', type=float, default=0.60, help='Soft agreement score needed to stop early after the initial dynamic view budget.')
    parser.add_argument('--early-stop-margin', dest='early_stop_margin', type=float, default=0.25, help='Probability margin needed to stop early after the initial dynamic view budget.')
    parser.add_argument('--enable-anchor-reservoir', dest='enable_anchor_reservoir', action='store_true', help='Enable ViT-only domain anchor reservoir and anchor-based logit correction.')
    parser.add_argument('--anchor-reservoir-size', dest='anchor_reservoir_size', type=int, default=4, help='Per-class capacity for the anchor reservoir.')
    parser.add_argument('--anchor-entropy-threshold', dest='anchor_entropy_threshold', type=float, default=0.25, help='Maximum normalized entropy for adding a sample to the anchor reservoir.')
    parser.add_argument('--anchor-alpha', dest='anchor_alpha', type=float, default=0.6, help='Strength of anchor-based logit correction.')
    parser.add_argument('--anchor-beta', dest='anchor_beta', type=float, default=1.5, help='Layer weighting temperature for anchor-based logit correction.')
    parser.add_argument('--mamba3-mode', dest='mamba3_mode', type=str, choices=['mamba3-trapezoid', 'mamba3-complex', 'mamba3-multislot'], default='mamba3-trapezoid', help='Mamba-3-inspired memory variant.')
    parser.add_argument('--mamba3-min-blend', dest='mamba3_min_blend', type=float, default=0.02, help='Minimum selective blend factor for Mamba-3-inspired memory.')
    parser.add_argument('--mamba3-max-blend', dest='mamba3_max_blend', type=float, default=0.35, help='Maximum selective blend factor for Mamba-3-inspired memory.')
    parser.add_argument('--mamba3-phase-strength', dest='mamba3_phase_strength', type=float, default=0.15, help='Phase rotation strength for mamba3-complex.')
    parser.add_argument('--mamba3-num-slots', dest='mamba3_num_slots', type=int, default=4, help='Number of per-class state slots for mamba3-multislot.')
    parser.add_argument('--mamba3-new-slot-threshold', dest='mamba3_new_slot_threshold', type=float, default=0.25, help='Similarity threshold for allocating a new slot in mamba3-multislot.')

    args = parser.parse_args()

    return args


def _build_memory(backend, params, clip_weights, image_features, include_prob_map=False, ssm_kwargs=None):
    if backend == 'cache':
        return CacheMemory(shot_capacity=params['shot_capacity'], include_prob_map=include_prob_map)

    if ssm_kwargs is None:
        ssm_kwargs = {}

    if backend == 'ssm-mamba3':
        return Mamba3Memory(
            num_classes=clip_weights.size(1),
            feature_dim=image_features.size(1),
            device=image_features.device,
            include_prob_map=include_prob_map,
            mode=ssm_kwargs['mamba3_mode'],
            min_blend=ssm_kwargs['mamba3_min_blend'],
            max_blend=ssm_kwargs['mamba3_max_blend'],
            phase_strength=ssm_kwargs['mamba3_phase_strength'],
            num_slots=ssm_kwargs['mamba3_num_slots'],
            new_slot_threshold=ssm_kwargs['mamba3_new_slot_threshold'],
        )

    return SSMemory(
        num_classes=clip_weights.size(1),
        feature_dim=image_features.size(1),
        device=image_features.device,
        include_prob_map=include_prob_map,
        correction_mode=ssm_kwargs['correction_mode'],
        kalman_q=ssm_kwargs['kalman_q'],
        kalman_r=ssm_kwargs['kalman_r'],
        kalman_q_min=ssm_kwargs['kalman_q_min'],
        kalman_q_max=ssm_kwargs['kalman_q_max'],
        kalman_r_min=ssm_kwargs['kalman_r_min'],
        kalman_r_max=ssm_kwargs['kalman_r_max'],
    )


def _compute_forgetting_curves(correct_history):
    if len(correct_history) == 0:
        return {
            'cumulative_accuracy': [],
            'forgetting_curve': [],
            'final_forgetting': 0.0,
            'mean_forgetting': 0.0,
        }

    cumulative_accuracy = []
    forgetting_curve = []
    running_best = 0.0
    correct_sum = 0.0

    for idx, is_correct in enumerate(correct_history, start=1):
        correct_sum += float(is_correct)
        current_acc = 100.0 * correct_sum / idx
        cumulative_accuracy.append(current_acc)
        running_best = max(running_best, current_acc)
        forgetting_curve.append(running_best - current_acc)

    return {
        'cumulative_accuracy': cumulative_accuracy,
        'forgetting_curve': forgetting_curve,
        'final_forgetting': forgetting_curve[-1],
        'mean_forgetting': sum(forgetting_curve) / len(forgetting_curve),
    }


def _sync_device(device):
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == 'mps' and hasattr(torch, 'mps'):
        torch.mps.synchronize()


def _get_peak_device_memory(device):
    if device.type == 'cuda' and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(), 'peak_cuda_memory_mb'

    if device.type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory'):
        return torch.mps.current_allocated_memory(), 'peak_mps_memory_mb'

    return 0, 'peak_device_memory_mb'


def run_test_tda(
    pos_cfg,
    neg_cfg,
    loader,
    clip_model,
    clip_weights,
    memory_type='cache',
    max_samples=None,
    enable_wandb=False,
    log_interval=1000,
    return_details=False,
    ssm_kwargs=None,
    consensus_kwargs=None,
    anchor_kwargs=None,
):
    device = next(clip_model.parameters()).device
    use_consensus = consensus_kwargs is not None and consensus_kwargs.get('n_views', 0) > 0
    if use_consensus:
        top_k = consensus_kwargs.get('top_k', 5)
        consensus_threshold = consensus_kwargs.get('consensus_threshold', 0.5)
        dynamic_view_budget = consensus_kwargs.get('dynamic_view_budget', False)
        dynamic_kwargs = consensus_kwargs.get('dynamic_kwargs', {})

    with torch.no_grad():
        pos_memory, neg_memory, accuracies = None, None, []
        anchor_memory = None
        correct_history = []
        step_times = []
        consensus_accept_count = 0
        consensus_total_count = 0
        soft_score_history = []
        update_weight_history = []
        accept_flag_history = []
        views_used_history = []
        view_savings_history = []
        early_stop_count = 0
        anchor_fill_ratio_history = []
        anchor_accept_flag_history = []
        anchor_correction_active_history = []
        anchor_correction_norm_history = []
        if device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        run_start = time.perf_counter()
        
        #Unpack all hyperparameters
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        anchor_enabled = anchor_kwargs is not None and anchor_kwargs.get('enabled', False)
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

        # Test-time adaptation
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            if max_samples is not None and i >= max_samples:
                break

            _sync_device(device)
            step_start = time.perf_counter()

            if use_consensus:
                if dynamic_view_budget:
                    image_features, clip_logits, loss, prob_map, pred, consensus_stats, visual_details = \
                        dynamic_multiview_consensus_logits(images, clip_model, clip_weights,
                                                           top_k=top_k,
                                                           consensus_threshold=consensus_threshold,
                                                           dynamic_kwargs=dynamic_kwargs,
                                                           device=device,
                                                           return_layer_cls=anchor_enabled)
                else:
                    image_features, clip_logits, loss, prob_map, pred, consensus_stats, visual_details = \
                        multiview_consensus_logits(images, clip_model, clip_weights,
                                                  top_k=top_k,
                                                  consensus_threshold=consensus_threshold,
                                                  device=device,
                                                  return_layer_cls=anchor_enabled)
                consensus_total_count += 1
                if consensus_stats['soft_accept']:
                    consensus_accept_count += 1
                soft_score_history.append(float(consensus_stats['soft_score']))
                update_weight_history.append(float(consensus_stats['update_weight']))
                accept_flag_history.append(1.0 if consensus_stats['soft_accept'] else 0.0)
                views_used_history.append(float(consensus_stats.get('num_views_used', 1)))
                view_savings_history.append(float(consensus_stats.get('view_savings', 0)))
                if dynamic_view_budget and consensus_stats.get('stop_reason', '') != 'max_budget':
                    early_stop_count += 1
            else:
                image_features, clip_logits, loss, prob_map, pred, visual_details = get_clip_logits_with_details(
                    images,
                    clip_model,
                    clip_weights,
                    device=device,
                    return_layer_cls=anchor_enabled,
                )
                consensus_stats = None

            target, prop_entropy = target.to(device), get_entropy(loss, clip_weights)

            if pos_enabled and pos_memory is None:
                pos_memory = _build_memory(memory_type, pos_params, clip_weights, image_features, include_prob_map=False, ssm_kwargs=ssm_kwargs)
            if neg_enabled and neg_memory is None:
                neg_memory = _build_memory(memory_type, neg_params, clip_weights, image_features, include_prob_map=True, ssm_kwargs=ssm_kwargs)
            if anchor_enabled and anchor_memory is None and visual_details.get('supports_anchor', False):
                anchor_memory = AnchorReservoir(
                    num_classes=clip_weights.size(1),
                    capacity=anchor_kwargs['capacity'],
                    device=device,
                    entropy_threshold=anchor_kwargs['entropy_threshold'],
                )

            update_weight = consensus_stats['update_weight'] if use_consensus else 1.0

            if pos_enabled:
                if memory_type == 'cache':
                    pos_memory.update(pred, image_features, float(loss), prob_map=None)
                else:
                    pos_memory.update(pred, image_features, prop_entropy, prob_map=None, update_weight=update_weight)

            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                if memory_type == 'cache':
                    neg_memory.update(pred, image_features, float(loss), prob_map=prob_map)
                else:
                    neg_memory.update(pred, image_features, prop_entropy, prob_map=prob_map, update_weight=update_weight)

            anchor_update_accepted = False
            if anchor_memory is not None:
                anchor_update_accepted = anchor_memory.update(
                    pred,
                    visual_details.get('layer_cls_tokens'),
                    prop_entropy,
                    update_weight=update_weight,
                )
                anchor_fill_ratio_history.append(anchor_memory.fill_ratio())
                anchor_accept_flag_history.append(1.0 if anchor_update_accepted else 0.0)
            else:
                anchor_fill_ratio_history.append(0.0)
                anchor_accept_flag_history.append(0.0)

            final_logits = clip_logits.clone()
            if pos_enabled and pos_memory is not None and not pos_memory.is_empty():
                final_logits += pos_memory.logits(image_features, pos_params['alpha'], pos_params['beta'], clip_weights)
            if neg_enabled and neg_memory is not None and not neg_memory.is_empty():
                final_logits -= neg_memory.logits(
                    image_features,
                    neg_params['alpha'],
                    neg_params['beta'],
                    clip_weights,
                    (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper'])
                )
            anchor_correction = None
            if anchor_memory is not None and not anchor_memory.is_empty():
                anchor_correction = anchor_memory.logits(
                    visual_details.get('layer_cls_tokens'),
                    anchor_kwargs['alpha'],
                    anchor_kwargs['beta'],
                ).to(final_logits.dtype)
                final_logits += anchor_correction

            anchor_correction_active_history.append(1.0 if anchor_correction is not None else 0.0)
            anchor_correction_norm_history.append(
                float(anchor_correction.norm().item()) if anchor_correction is not None else 0.0
            )

                
            acc = cls_acc(final_logits, target)  
            accuracies.append(acc)
            pred_label = int(final_logits.topk(1, 1, True, True)[1].t()[0])
            correct_history.append(1.0 if pred_label == int(target.item()) else 0.0)

            if enable_wandb:
                import wandb
                wandb.log({"Averaged test accuracy": sum(accuracies)/len(accuracies)}, commit=True)

            _sync_device(device)
            step_times.append(time.perf_counter() - step_start)

            if i % log_interval == 0:
                print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))

        total_runtime = time.perf_counter() - run_start
        peak_device_mem, peak_mem_key = _get_peak_device_memory(device)
        adapter_memory_bytes = 0
        if pos_memory is not None:
            adapter_memory_bytes += pos_memory.memory_bytes()
        if neg_memory is not None:
            adapter_memory_bytes += neg_memory.memory_bytes()
        if anchor_memory is not None:
            adapter_memory_bytes += anchor_memory.memory_bytes()

        avg_acc = sum(accuracies)/len(accuracies) if len(accuracies) > 0 else 0.0
        print("---- TDA's test accuracy: {:.2f}. ----\n".format(avg_acc))

        if not return_details:
            return avg_acc

        forgetting_metrics = _compute_forgetting_curves(correct_history)
        details = {
            'accuracy': avg_acc,
            'num_samples': len(accuracies),
            'cumulative_accuracy': forgetting_metrics['cumulative_accuracy'],
            'forgetting_curve': forgetting_metrics['forgetting_curve'],
            'final_forgetting': forgetting_metrics['final_forgetting'],
            'mean_forgetting': forgetting_metrics['mean_forgetting'],
            'avg_step_time_ms': (sum(step_times) / len(step_times) * 1000.0) if len(step_times) > 0 else 0.0,
            'total_runtime_s': total_runtime,
            peak_mem_key: peak_device_mem / (1024.0 * 1024.0),
            'adapter_memory_mb': adapter_memory_bytes / (1024.0 * 1024.0),
        }
        if use_consensus and consensus_total_count > 0:
            details['consensus_accept_rate'] = consensus_accept_count / consensus_total_count
            soft_score_prefix = np.cumsum(np.array(soft_score_history, dtype=np.float64))
            update_weight_prefix = np.cumsum(np.array(update_weight_history, dtype=np.float64))
            accept_prefix = np.cumsum(np.array(accept_flag_history, dtype=np.float64))
            steps = np.arange(1, len(soft_score_history) + 1, dtype=np.float64)
            details['avg_soft_score'] = float(soft_score_prefix[-1] / steps[-1])
            details['avg_update_weight'] = float(update_weight_prefix[-1] / steps[-1])
            details['soft_score_curve'] = (soft_score_prefix / steps).tolist()
            details['update_weight_curve'] = (update_weight_prefix / steps).tolist()
            details['consensus_accept_rate_curve'] = (accept_prefix / steps).tolist()
            if len(views_used_history) > 0:
                views_used_prefix = np.cumsum(np.array(views_used_history, dtype=np.float64))
                view_savings_prefix = np.cumsum(np.array(view_savings_history, dtype=np.float64))
                details['avg_views_used'] = float(views_used_prefix[-1] / steps[-1])
                details['avg_view_savings'] = float(view_savings_prefix[-1] / steps[-1])
                details['avg_views_used_curve'] = (views_used_prefix / steps).tolist()
                details['avg_view_savings_curve'] = (view_savings_prefix / steps).tolist()
                if dynamic_view_budget:
                    details['early_stop_rate'] = early_stop_count / consensus_total_count
        if anchor_memory is not None:
            anchor_stats = anchor_memory.stats()
            details['anchor_update_accept_count'] = anchor_stats['accepted_updates']
            details['anchor_update_attempt_count'] = anchor_stats['total_update_attempts']
            details['anchor_update_accept_rate'] = anchor_stats['accept_rate']
            details['anchor_fill_ratio'] = anchor_stats['fill_ratio']
            if len(anchor_fill_ratio_history) > 0:
                anchor_fill_prefix = np.cumsum(np.array(anchor_fill_ratio_history, dtype=np.float64))
                anchor_accept_prefix = np.cumsum(np.array(anchor_accept_flag_history, dtype=np.float64))
                anchor_active_prefix = np.cumsum(np.array(anchor_correction_active_history, dtype=np.float64))
                anchor_norm_prefix = np.cumsum(np.array(anchor_correction_norm_history, dtype=np.float64))
                anchor_steps = np.arange(1, len(anchor_fill_ratio_history) + 1, dtype=np.float64)
                details['avg_anchor_fill_ratio'] = float(anchor_fill_prefix[-1] / anchor_steps[-1])
                details['avg_anchor_correction_active_rate'] = float(anchor_active_prefix[-1] / anchor_steps[-1])
                details['avg_anchor_correction_norm'] = float(anchor_norm_prefix[-1] / anchor_steps[-1])
                details['anchor_update_accept_rate_curve'] = (anchor_accept_prefix / anchor_steps).tolist()
                details['anchor_fill_ratio_curve'] = (anchor_fill_prefix / anchor_steps).tolist()
                details['anchor_correction_active_rate_curve'] = (anchor_active_prefix / anchor_steps).tolist()
                details['anchor_correction_norm_curve'] = (anchor_norm_prefix / anchor_steps).tolist()
        if memory_type == 'ssm-mamba3' and pos_memory is not None:
            mamba3_stats = pos_memory.stats()
            details['mamba3_update_accept_count'] = mamba3_stats['accepted_updates']
            details['mamba3_update_attempt_count'] = mamba3_stats['total_update_attempts']
            details['mamba3_update_accept_rate'] = mamba3_stats['accept_rate']
            details['mamba3_active_slot_ratio'] = mamba3_stats['active_slot_ratio']
            details['mamba3_avg_slot_utilization'] = mamba3_stats['avg_slot_utilization']
            details['mamba3_avg_phase_magnitude'] = mamba3_stats['avg_phase_magnitude']
            details['mamba3_avg_alpha'] = mamba3_stats['avg_alpha']
            details['mamba3_avg_beta'] = mamba3_stats['avg_beta']
            details['mamba3_avg_gamma'] = mamba3_stats['avg_gamma']
        return details



def main():
    global wandb
    args = get_arguments()
    config_path = args.config

    if args.wandb:
        if importlib.util.find_spec('wandb') is None:
            raise ImportError('wandb is not installed. Install it or remove --wandb-log.')
        wandb = importlib.import_module('wandb')

    # Initialize CLIP model
    device = resolve_device(args.device)
    clip_model, preprocess = clip.load(args.backbone, device=device)
    if device.type == 'mps':
        clip_model.float()
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"

    ssm_kwargs = {
        'correction_mode': args.ssm_correction_mode,
        'kalman_q': args.kalman_q,
        'kalman_r': args.kalman_r,
        'kalman_q_min': args.kalman_q_min,
        'kalman_q_max': args.kalman_q_max,
        'kalman_r_min': args.kalman_r_min,
        'kalman_r_max': args.kalman_r_max,
        'mamba3_mode': args.mamba3_mode,
        'mamba3_min_blend': args.mamba3_min_blend,
        'mamba3_max_blend': args.mamba3_max_blend,
        'mamba3_phase_strength': args.mamba3_phase_strength,
        'mamba3_num_slots': args.mamba3_num_slots,
        'mamba3_new_slot_threshold': args.mamba3_new_slot_threshold,
    }
    anchor_kwargs = {
        'enabled': args.enable_anchor_reservoir,
        'capacity': args.anchor_reservoir_size,
        'entropy_threshold': args.anchor_entropy_threshold,
        'alpha': args.anchor_alpha,
        'beta': args.anchor_beta,
    }

    consensus_kwargs = None
    if args.n_views > 0:
        dynamic_max_aug_views = args.n_views
        if args.dynamic_view_budget:
            dynamic_max_aug_views = min(args.n_views, max(0, args.dynamic_max_views - 1))
        consensus_kwargs = {
            'n_views': dynamic_max_aug_views,
            'top_k': args.top_k,
            'consensus_threshold': args.consensus_threshold,
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
    
    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        test_loader, classnames, template = build_test_data_loader(
            dataset_name, args.data_root, preprocess,
            n_views=consensus_kwargs['n_views'] if consensus_kwargs is not None else args.n_views)
        clip_weights = clip_classifier(classnames, template, clip_model, device=device)

        if args.wandb:
            run_name = f"{dataset_name}"
            run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name, name=run_name)

        acc = run_test_tda(
            cfg['positive'],
            cfg['negative'],
            test_loader,
            clip_model,
            clip_weights,
            memory_type=args.memory_type,
            max_samples=args.max_samples,
            enable_wandb=args.wandb,
            ssm_kwargs=ssm_kwargs,
            consensus_kwargs=consensus_kwargs,
            anchor_kwargs=anchor_kwargs,
        )

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()

if __name__ == "__main__":
    main()
