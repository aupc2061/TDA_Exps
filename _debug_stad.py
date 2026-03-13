"""Debug: check STAD-vMF across kappa settings and compare with TDA."""
import os
import torch
import clip
import random
from utils import resolve_device, build_test_data_loader, clip_classifier
from stad_baseline import STADvMF
import torch.nn.functional as F

CACHE_PATH = '/tmp/stad_debug_feats.pt'

def main():
    device = resolve_device('auto')
    model, preprocess = clip.load('RN50', device=device)
    model.float().eval()
    random.seed(1)
    torch.manual_seed(1)

    loader, classnames, template = build_test_data_loader('oxford_pets', '.', preprocess)
    cw = clip_classifier(classnames, template, model, device=device)
    print(f'Classes: {len(classnames)}, clip_weights: {cw.shape}')

    # Cache features to avoid repeated dataloader issues
    if os.path.exists(CACHE_PATH):
        print('Loading cached features...')
        cached = torch.load(CACHE_PATH, map_location=device, weights_only=True)
        all_feats = cached['feats']
        all_labels = cached['labels']
    else:
        print('Extracting features (will cache for reuse)...')
        # Use num_workers=0 to avoid multiprocessing issues on macOS
        from torch.utils.data import DataLoader
        dataset = loader.dataset
        single_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        feats_list, labels_list = [], []
        for images, target in single_loader:
            if len(feats_list) >= 500:
                break
            with torch.no_grad():
                images = images.to(device)
                f = model.encode_image(images).float()
                f = f / f.norm(dim=-1, keepdim=True)
                feats_list.append(f)
                labels_list.append(target.item())
        all_feats = torch.cat(feats_list, dim=0)
        all_labels = torch.tensor(labels_list, device=device)
        torch.save({'feats': all_feats.cpu(), 'labels': all_labels.cpu()}, CACHE_PATH)
        print(f'Cached {len(all_feats)} features to {CACHE_PATH}')

    all_feats = all_feats.to(device)
    all_labels = all_labels.to(device)

    # 1) Zero-shot accuracy
    logits_zs = all_feats @ cw
    preds_zs = logits_zs.argmax(dim=1)
    zs_acc = (preds_zs == all_labels).float().mean().item() * 100
    print(f'Zero-shot accuracy (500 samples): {zs_acc:.1f}%')

    # 2) Grid search over kappa
    print('\n--- STAD-vMF kappa grid search (bs=64, 500 samples) ---')
    for kt in [100, 1000, 10000]:
        for ke in [100, 1000, 10000]:
            stad = STADvMF(
                num_classes=cw.size(1), feature_dim=cw.size(0),
                source_prototypes=cw, device=device,
                kappa_trans=float(kt), kappa_ems=float(ke),
                window_size=3, n_em_iters=5,
            )
            correct = 0
            bs = 64
            for start in range(0, len(all_feats), bs):
                end = min(start + bs, len(all_feats))
                H = all_feats[start:end]
                stad.update(H)
                logits = stad.predict(H)
                preds = logits.argmax(dim=1)
                correct += (preds == all_labels[start:end]).sum().item()
            acc = 100 * correct / len(all_feats)
            print(f'  kt={kt:6d} ke={ke:6d} -> {acc:.1f}%')

    # 3) Also test with n_em_iters=1 (less aggressive adaptation)
    print('\n--- STAD-vMF with n_em_iters=1 (bs=64, 500 samples) ---')
    for kt in [1000, 10000]:
        for ke in [1000, 10000]:
            stad = STADvMF(
                num_classes=cw.size(1), feature_dim=cw.size(0),
                source_prototypes=cw, device=device,
                kappa_trans=float(kt), kappa_ems=float(ke),
                window_size=3, n_em_iters=1,
            )
            correct = 0
            bs = 64
            for start in range(0, len(all_feats), bs):
                end = min(start + bs, len(all_feats))
                H = all_feats[start:end]
                stad.update(H)
                logits = stad.predict(H)
                preds = logits.argmax(dim=1)
                correct += (preds == all_labels[start:end]).sum().item()
            acc = 100 * correct / len(all_feats)
            print(f'  kt={kt:6d} ke={ke:6d} iter=1 -> {acc:.1f}%')

    print('\nDone.')


if __name__ == '__main__':
    main()
