"""
5-way comparison: CLIP zero-shot vs TDA SSM variants vs STAD standalone vs STAD-as-correction.

Usage:
  python compare_stad_tda.py --dataset ucf101 --n-samples 2000
  python compare_stad_tda.py --dataset oxford_pets --n-samples 500
"""
import os
import csv
import argparse
import torch
import clip
import random
import yaml
import torch.nn.functional as F
from utils import resolve_device, build_test_data_loader, clip_classifier
from stad_baseline import STADvMF
from memory import SSMemory

CACHE_DIR = '/tmp/tda_comparison'


# ── Feature extraction / caching ────────────────────────────────────

def extract_or_load(dataset_name, model, preprocess, device, n_samples, data_root='./dataset/'):
    """Extract CLIP features + logits once and cache to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f'{dataset_name}_{n_samples}.pt')

    if os.path.exists(cache_path):
        print(f'  Loading cached features from {cache_path}')
        data = torch.load(cache_path, map_location='cpu', weights_only=True)
        return data['feats'].to(device), data['labels'].to(device), data['logits'].to(device)

    print(f'  Extracting {n_samples} features for {dataset_name}...')
    loader, classnames, template = build_test_data_loader(dataset_name, data_root, preprocess)
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
            logits_list.append((100.0 * f @ cw).cpu())

    all_feats = torch.cat(feats, dim=0)
    all_labels = torch.tensor(labels)
    all_logits = torch.cat(logits_list, dim=0)

    torch.save({'feats': all_feats, 'labels': all_labels, 'logits': all_logits}, cache_path)
    print(f'  Cached {len(all_feats)} features')
    return all_feats.to(device), all_labels.to(device), all_logits.to(device)


# ── Evaluation functions ────────────────────────────────────────────

def eval_zero_shot(all_feats, all_labels, cw):
    """CLIP zero-shot: logits = 100 * features @ clip_weights."""
    logits = 100.0 * all_feats @ cw
    preds = logits.argmax(dim=1)
    return (preds == all_labels).float().mean().item() * 100


def eval_tda_ssm(all_feats, all_labels, all_logits, cw, device,
                 mode='heuristic', alpha=1.0, beta=5.5):
    """TDA SSM correction on top of CLIP logits (our method).

    final_logits = clip_logits + ssm.logits(feat, alpha, beta, cw)
    The SSM stores per-class state vectors updated via selective
    gating (Mamba-style), and produces a correction signal.
    """
    K = cw.size(1)
    D = cw.size(0)

    ssm = SSMemory(num_classes=K, feature_dim=D, device=device,
                   correction_mode=mode)

    correct = 0
    for i in range(len(all_feats)):
        feat = all_feats[i:i+1]
        label = all_labels[i]
        clip_logit = all_logits[i:i+1]

        # Normalised entropy for SSM gating
        ent = (-clip_logit.softmax(1) * clip_logit.log_softmax(1)).sum(1)
        prop_entropy = float(ent / torch.log(torch.tensor(float(K))))

        pred = clip_logit.argmax(dim=1).item()
        ssm.update(pred, feat, prop_entropy)

        final_logits = clip_logit.clone()
        if not ssm.is_empty():
            final_logits = final_logits + ssm.logits(feat, alpha, beta, cw)

        correct += int(final_logits.argmax(dim=1).item() == label.item())

    return 100.0 * correct / len(all_feats)


def eval_stad_standalone(all_feats, all_labels, cw, device,
                         kt=1000, ke=1000, bs=64, n_iters=5):
    """STAD-vMF standalone: replaces classifier entirely with EM-adapted prototypes.

    logits = features @ rho  (cosine sim to adapted prototypes)
    """
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
        preds = stad.predict(H).argmax(dim=1)
        correct += (preds == all_labels[start:end]).sum().item()
    return 100.0 * correct / len(all_feats)


def eval_stad_as_correction(all_feats, all_labels, all_logits, cw, device,
                            kt=1000, ke=1000, bs=64, n_iters=5, alpha=0.05):
    """STAD-vMF as additive correction on top of CLIP logits (hybrid).

    Mathematically equivalent to:
      final = 100 * ((1 - alpha) * features @ cw  +  alpha * features @ rho)

    i.e. linear interpolation between zero-shot logits and STAD logits.
    At alpha=0: pure zero-shot.  At alpha=1: pure STAD (standalone).
    """
    stad = STADvMF(
        num_classes=cw.size(1), feature_dim=cw.size(0),
        source_prototypes=cw, device=device,
        kappa_trans=float(kt), kappa_ems=float(ke),
        window_size=3, n_em_iters=n_iters,
    )
    # Source prototypes (cw is already column-normalised from clip_classifier)
    source_protos = F.normalize(cw, dim=0)  # no-op but explicit

    correct = 0
    for start in range(0, len(all_feats), bs):
        end = min(start + bs, len(all_feats))
        H = all_feats[start:end]

        stad.update(H)

        # correction = cosine_sim(H, adapted) - cosine_sim(H, source)
        correction = H @ stad.rho - H @ source_protos  # (B, K)

        # Scale to CLIP logit range and blend
        final_logits = all_logits[start:end] + alpha * 100.0 * correction

        correct += (final_logits.argmax(dim=1) == all_labels[start:end]).sum().item()

    return 100.0 * correct / len(all_feats)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ucf101')
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--data-root', default='./dataset/',
                        help='Root dir containing dataset folders')
    parser.add_argument('--output', default=None, help='CSV output path (auto-generated if omitted)')
    args = parser.parse_args()

    device = resolve_device('auto')
    model, preprocess = clip.load('RN50', device=device)
    model.float().eval()
    random.seed(1)
    torch.manual_seed(1)

    dataset_name = args.dataset
    n_samples = args.n_samples

    # Load per-dataset config for TDA alpha/beta
    cfg_path = f'configs/{dataset_name}.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    tda_alpha = cfg['positive']['alpha']
    tda_beta = cfg['positive']['beta']

    print(f'\n{"="*60}')
    print(f'  Dataset: {dataset_name}  |  Samples: {n_samples}')
    print(f'  TDA config: alpha={tda_alpha}, beta={tda_beta}')
    print(f'{"="*60}')

    data_root = args.data_root
    loader, classnames, template = build_test_data_loader(dataset_name, data_root, preprocess)
    cw = clip_classifier(classnames, template, model, device=device)
    print(f'  Classes: {len(classnames)}, Feature dim: {cw.size(0)}')

    all_feats, all_labels, all_logits = extract_or_load(
        dataset_name, model, preprocess, device, n_samples=n_samples,
        data_root=data_root,
    )
    print(f'  Loaded {len(all_feats)} samples\n')

    # Collect results
    rows = []

    def record(method, accuracy, **kwargs):
        row = {'dataset': dataset_name, 'method': method,
               'n_samples': len(all_feats), 'accuracy': round(accuracy, 2)}
        row.update(kwargs)
        rows.append(row)
        print(f'  {method:45s}  {accuracy:.2f}%')

    # A) CLIP zero-shot
    record('clip_zero_shot', eval_zero_shot(all_feats, all_labels, cw))

    # B) TDA SSM (our method) – heuristic and vmf-fixed
    for mode in ['heuristic', 'vmf-fixed']:
        acc = eval_tda_ssm(all_feats, all_labels, all_logits, cw, device,
                           mode=mode, alpha=tda_alpha, beta=tda_beta)
        record(f'tda_ssm_{mode}', acc)

    # C) STAD standalone (classifier replacement)
    acc = eval_stad_standalone(all_feats, all_labels, cw, device,
                               kt=1000, ke=1000)
    record('stad_standalone', acc)

    # D) STAD as correction on CLIP (best alpha sweep)
    best_alpha, best_acc = 0.0, 0.0
    print('\n  --- STAD-as-correction alpha sweep ---')
    for alpha in [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]:
        acc = eval_stad_as_correction(all_feats, all_labels, all_logits, cw, device,
                                      kt=1000, ke=1000, alpha=alpha)
        print(f'     alpha={alpha:.2f} -> {acc:.2f}%')
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
    record(f'stad_correction_best(alpha={best_alpha})', best_acc)

    # CSV output
    out_path = args.output or f'results/{dataset_name}_stad_comparison.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ['dataset', 'method', 'n_samples', 'accuracy']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n  Results saved to {out_path}')


if __name__ == '__main__':
    main()
