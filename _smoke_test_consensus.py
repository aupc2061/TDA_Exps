"""Quick smoke test for the consensus gating pipeline."""
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import torch
import clip
from utils import (
    build_test_data_loader, clip_classifier,
    multiview_consensus_logits, resolve_device,
)


def main():
    device = resolve_device('auto')
    print(f'Device: {device}')
    clip_model, preprocess = clip.load('RN50', device=device)
    if device.type == 'mps':
        clip_model.float()
    clip_model.eval()

    # Build data loader with consensus views
    test_loader, classnames, template = build_test_data_loader(
        'oxford_pets', '.', preprocess, shuffle=False, n_views=8)
    clip_weights = clip_classifier(classnames, template, clip_model, device=device)
    print(f'Classes: {len(classnames)}, clip_weights: {clip_weights.shape}')

    # Test a few batches
    for i, (images, target) in enumerate(test_loader):
        if i >= 5:
            break
        print(f'\n--- Sample {i} ---')
        print(f'images type: {type(images)}, len: {len(images) if isinstance(images, list) else "N/A"}')
        if isinstance(images, list):
            print(f'  view shapes: {[v.shape for v in images[:3]]} ...')

        feat, logits, loss, prob_map, pred, consensus = multiview_consensus_logits(
            images, clip_model, clip_weights,
            top_k=5, consensus_threshold=0.5, device=device)

        print(f'features: {feat.shape}, logits: {logits.shape}')
        print(f'pred: {pred} ({classnames[pred]}), target: {target.item()} ({classnames[target.item()]})')
        print(f'consensus: {consensus}, loss: {loss.item():.4f}')

    print('\nSmoke test PASSED')


if __name__ == '__main__':
    main()
