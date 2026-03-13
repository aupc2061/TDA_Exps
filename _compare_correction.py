"""
3-way comparison: Our vMF correction vs STAD standalone vs STAD-as-correction.

Compares:
  A) CLIP zero-shot (baseline)
  B) TDA SSM vMF correction on top of CLIP (our method)
  C) STAD-vMF standalone (classifier replacement)
  D) STAD-vMF as additive correction on top of CLIP (hybrid)
"""
import os
import torch
import clip
import random
import torch.nn.functional as F
from utils import resolve_device, build_test_data_loader, clip_classifier, get_entropy
from stad_baseline import STADvMF
from memory import SSMemory

CACHE_DIR = '/tmp/tda_comparison'
# Also check for existing cache from debug script
LEGACY_CACHE = '/tmp/stad_debug_feats.pt'


def extract_or_load(dataset_name, model, preprocess, device, n_samples=500):
    """Extract CLIP features once and cache them."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f'{dataset_name}_{n_samples}.pt')

    if os.path.exists(cache_path):
        print(f'Loading cached features from {cache_path}')
        data = torch.load(cache_path, map_location='cpu', weights_only=True)
        return data['feats'].to(device), data['labels'].to(device), data['logits'].to(device)

    # Check legacy cache (from _debug_stad.py) – has feats+labels but no logits
    if dataset_name == 'oxford_pets' and n_samples == 500 and os.path.exists(LEGACY_CACHE):
        print(f'Loading legacy cache from {LEGACY_CACHE} (will add logits)')
        data = torch.load(LEGACY_CACHE, map_location='cpu', weights_only=True)
        feats = data['feats'].to(device)
        labs = data['labels'].to(device)
        loader_tmp, cn_tmp, tmpl_tmp = build_test_data_loader(dataset_name, '.', preprocess)
        cw_tmp = clip_classifier(cn_tmp, tmpl_tmp, model, device=device)
        logits = 100.0 * feats @ cw_tmp
        # Save to new cache with logits
        torch.save({'feats': feats.cpu(), 'labels': labs.cpu(), 'logits': logits.cpu()}, cache_path)
        return feats, labs, logits

    print(f'Extracting {n_samples} features for {dataset_name}...')
    loader, classnames, template = build_test_data_loader(dataset_name, '.', preprocess)
    cw = clip_classifier(classnames, template, model, device=device)

    from torch.utils.data import DataLoader
    single_loader = DataLoader(loader.dataset, batch_size=1, shuffle=False, num_workers=0)

    feats, labels, logits_list = [], [], []
    for images, target in single_loader:
        if len(feats) >= n_samples:
            break
        with torch.no_grad():
            images = images.to(device)
            f = model.encode_image(images).float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
            labels.append(target.item())
            # Store raw clip logits (100 * f @ cw matches get_clip_logits)
            logit = 100.0 * f @ cw
            logits_list.append(logit.cpu())

    all_feats = torch.cat(feats, dim=0)
    all_labels = torch.tensor(labels)
    all_logits = torch.cat(logits_list, dim=0)

    torch.save({'feats': all_feats, 'labels': all_labels, 'logits': all_logits}, cache_path)
    print(f'Cached {len(all_feats)} features to {cache_path}')
    return all_feats.to(device), all_labels.to(device), all_logits.to(device)


def eval_zero_shot(all_feats, all_labels, cw):
    """A) CLIP zero-shot."""
    logits = 100.0 * all_feats @ cw
    preds = logits.argmax(dim=1)
    return (preds == all_labels).float().mean().item() * 100


def eval_tda_ssm(all_feats, all_labels, all_logits, cw, device, mode='heuristic'):
    """B) TDA SSM correction on top of CLIP logits (our method)."""
    K = cw.size(1)
    D = cw.size(0)

    ssm = SSMemory(
        num_classes=K, feature_dim=D, device=device,
        correction_mode=mode,
    )

    # Config params (from oxford_pets.yaml defaults)
    alpha_pos = 1.0
    beta_pos = 5.5

    correct = 0
    for i in range(len(all_feats)):
        feat = all_feats[i:i+1]  # (1, D)
        label = all_labels[i]
        clip_logit = all_logits[i:i+1]  # (1, K)

        # Compute entropy for SSM update
        loss = ((-clip_logit.softmax(1) * clip_logit.log_softmax(1)).sum(1))
        prop_entropy = float(loss / torch.log(torch.tensor(float(K))))

        pred = clip_logit.argmax(dim=1).item()

        # Update SSM
        ssm.update(pred, feat, prop_entropy)

        # Compute corrected logits
        final_logits = clip_logit.clone()
        if not ssm.is_empty():
            final_logits = final_logits + ssm.logits(feat, alpha_pos, beta_pos, cw)

        pred_label = final_logits.argmax(dim=1).item()
        correct += int(pred_label == label.item())

    return 100.0 * correct / len(all_feats)


def eval_stad_standalone(all_feats, all_labels, cw, device, kt=1000, ke=1000, bs=64, n_iters=5):
    """C) STAD-vMF standalone (classifier replacement)."""
    stad = STADvMF(
        num_classes=cw.size(1), feature_dim=cw.size(0),
        source_prototypes=cw, device=device,
        kappa_trans=float(kt), kappa_ems=float(ke),
        window_size=3, n_em_iters=n_iters,
    )
    correct = 0
    for start in range(0, len(all_feats), bs):
        end = min(start + bs, len(all_feats))
        H = all_feats[start:end]
        stad.update(H)
        logits = stad.predict(H)
        preds = logits.argmax(dim=1)
        correct += (preds == all_labels[start:end]).sum().item()
    return 100.0 * correct / len(all_feats)


def eval_stad_as_correction(all_feats, all_labels, all_logits, cw, device,
                            kt=1000, ke=1000, bs=64, n_iters=5, alpha=1.0):
    """D) STAD-vMF prototypes as additive correction on top of CLIP logits.

    correction = features @ (adapted_prototypes - source_prototypes)
    final_logits = clip_logits + alpha * correction
    """
    source_protos = F.normalize(cw, dim=0)  # (D, K) – same as STAD's mu0

    stad = STADvMF(
        num_classes=cw.size(1), feature_dim=cw.size(0),
        source_prototypes=cw, device=device,
        kappa_trans=float(kt), kappa_ems=float(ke),
        window_size=3, n_em_iters=n_iters,
    )

    correct = 0
    idx = 0
    for start in range(0, len(all_feats), bs):
        end = min(start + bs, len(all_feats))
        H = all_feats[start:end]

        # Update STAD prototypes
        stad.update(H)

        # Correction = cosine sim to adapted - cosine sim to source
        adapted_logits = H @ stad.rho       # (B, K)
        source_logits = H @ source_protos   # (B, K)
        correction = adapted_logits - source_logits  # (B, K)

        # Add correction to CLIP logits
        clip_logit = all_logits[start:end]  # (B, K)
        # Scale correction to match CLIP logit scale (100x)
        final_logits = clip_logit + alpha * 100.0 * correction

        preds = final_logits.argmax(dim=1)
        correct += (preds == all_labels[start:end]).sum().item()
        idx += end - start

    return 100.0 * correct / len(all_feats)


def main():
    device = resolve_device('auto')
    model, preprocess = clip.load('RN50', device=device)
    model.float().eval()
    random.seed(1)
    torch.manual_seed(1)

    for dataset_name in ['oxford_pets']:
        print(f'\n{"="*60}')
        print(f'  Dataset: {dataset_name}')
        print(f'{"="*60}')

        loader, classnames, template = build_test_data_loader(dataset_name, '.', preprocess)
        cw = clip_classifier(classnames, template, model, device=device)
        print(f'  Classes: {len(classnames)}, D: {cw.size(0)}')

        n_samples = 500
        all_feats, all_labels, all_logits = extract_or_load(
            dataset_name, model, preprocess, device, n_samples=n_samples
        )
        print(f'  Samples: {len(all_feats)}')

        # A) Zero-shot
        zs = eval_zero_shot(all_feats, all_labels, cw)
        print(f'\n  A) CLIP zero-shot:             {zs:.1f}%')

        # B) TDA SSM (our vMF correction)
        for mode in ['heuristic', 'vmf-fixed']:
            tda = eval_tda_ssm(all_feats, all_labels, all_logits, cw, device, mode=mode)
            print(f'  B) TDA SSM ({mode:12s}):   {tda:.1f}%')

        # C) STAD standalone
        for kt, ke in [(1000, 1000), (10000, 10000)]:
            stad = eval_stad_standalone(all_feats, all_labels, cw, device, kt=kt, ke=ke)
            print(f'  C) STAD standalone (kt={kt}, ke={ke}): {stad:.1f}%')

        # D) STAD as correction on top of CLIP
        print(f'\n  D) STAD-vMF as correction on CLIP logits:')
        for kt, ke in [(1000, 1000), (10000, 10000)]:
            for alpha in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
                hybrid = eval_stad_as_correction(
                    all_feats, all_labels, all_logits, cw, device,
                    kt=kt, ke=ke, alpha=alpha,
                )
                print(f'     kt={kt:5d} ke={ke:5d} alpha={alpha:.2f} -> {hybrid:.1f}%')

        # D') STAD correction with fewer EM iters
        print(f'\n  D\') STAD-vMF correction, n_iters=1:')
        for kt, ke in [(1000, 1000), (10000, 10000)]:
            for alpha in [0.01, 0.05, 0.1, 0.15, 0.2]:
                hybrid = eval_stad_as_correction(
                    all_feats, all_labels, all_logits, cw, device,
                    kt=kt, ke=ke, n_iters=1, alpha=alpha,
                )
                print(f'     kt={kt:5d} ke={ke:5d} iter=1 alpha={alpha:.2f} -> {hybrid:.1f}%')

    print('\nDone.')


if __name__ == '__main__':
    main()
