import random
import argparse
from tqdm import tqdm
from datetime import datetime
import time
import importlib

import torch

import clip
from utils import *
from memory import CacheMemory, SSMemory

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
    parser.add_argument('--memory-type', dest='memory_type', type=str, choices=['cache', 'ssm'], default='cache', help='Adapter memory backend: original finite cache or SSM memory. Default is cache.')
    parser.add_argument('--max-samples', dest='max_samples', type=int, default=None, help='Optional maximum stream length per dataset.')

    args = parser.parse_args()

    return args


def _build_memory(backend, params, clip_weights, image_features, include_prob_map=False):
    if backend == 'cache':
        return CacheMemory(shot_capacity=params['shot_capacity'], include_prob_map=include_prob_map)

    return SSMemory(
        num_classes=clip_weights.size(1),
        feature_dim=image_features.size(1),
        device=image_features.device,
        include_prob_map=include_prob_map,
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
):
    device = next(clip_model.parameters()).device

    with torch.no_grad():
        pos_memory, neg_memory, accuracies = None, None, []
        correct_history = []
        step_times = []
        if device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        run_start = time.perf_counter()
        
        #Unpack all hyperparameters
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
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

            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images ,clip_model, clip_weights, device=device)
            target, prop_entropy = target.to(device), get_entropy(loss, clip_weights)

            if pos_enabled and pos_memory is None:
                pos_memory = _build_memory(memory_type, pos_params, clip_weights, image_features, include_prob_map=False)
            if neg_enabled and neg_memory is None:
                neg_memory = _build_memory(memory_type, neg_params, clip_weights, image_features, include_prob_map=True)

            if pos_enabled:
                pos_memory.update(pred, image_features, float(loss), prob_map=None) if memory_type == 'cache' else pos_memory.update(pred, image_features, prop_entropy, prob_map=None)

            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                neg_memory.update(pred, image_features, float(loss), prob_map=prob_map) if memory_type == 'cache' else neg_memory.update(pred, image_features, prop_entropy, prob_map=prob_map)

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

        avg_acc = sum(accuracies)/len(accuracies) if len(accuracies) > 0 else 0.0
        print("---- TDA's test accuracy: {:.2f}. ----\n".format(avg_acc))

        if not return_details:
            return avg_acc

        forgetting_metrics = _compute_forgetting_curves(correct_history)
        return {
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
    
    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
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
        )

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()

if __name__ == "__main__":
    main()