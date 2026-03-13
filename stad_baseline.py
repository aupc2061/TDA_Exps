"""
STAD-vMF baseline implementation.

Reproduces the core algorithm from:
  "Temporal Test-Time Adaptation with State-Space Models"
  Schirmer, Zhang, Nalisnick (TMLR 2024)  --  arXiv:2407.12492

STAD-vMF models each class prototype as a latent direction on the unit
hypersphere.  At every time-step a variational EM update refines the
prototypes given a new batch of normalised CLIP features.  The adapted
classification head is  softmax(W_t^T h) .

This file is self-contained: it implements the STAD-vMF model, an
evaluation loop that uses the same CLIP backbone / dataset loaders as
TDA, and a CLI for quick benchmarking.
"""

import math
import time
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

import clip
from utils import (
    resolve_device,
    get_config_file,
    build_test_data_loader,
    clip_classifier,
    cls_acc,
)


# ── vMF helpers ──────────────────────────────────────────────────────

def _log_bessel_ratio(D, kappa):
    """Approximate  A_D(kappa) = I_{D/2}(kappa) / I_{D/2-1}(kappa).

    Uses the reciprocal form  A_D(k) = 2k / (2k + D - 1)  which is
    bounded in (0, 1) for all  k > 0  and all  D >= 2.  This avoids
    the classical first-order expansion  1 - (D-1)/(2k)  that goes
    negative when  k < (D-1)/2  (e.g. D=1024, k=100).
    """
    two_k = 2.0 * kappa
    approx = two_k / (two_k + D - 1.0 + 1e-8)
    return approx.clamp(min=1e-6, max=1.0 - 1e-6)


def _estimate_kappa(r_bar, D):
    """MLE estimate of kappa from mean resultant length r_bar (Banerjee 2005)."""
    return (r_bar * D - r_bar ** 3) / (1.0 - r_bar ** 2 + 1e-8)


# ── STAD-vMF model ──────────────────────────────────────────────────

class STADvMF:
    """Variational-EM STAD-vMF for test-time adaptation.

    Maintains per-class prototypes rho_{t,k} on the unit sphere and
    updates them with each incoming batch using the algorithm from
    Schirmer et al. (Algorithm 3 in the paper).
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        source_prototypes: torch.Tensor,   # (D, K) – column = class
        device: torch.device,
        kappa_trans: float = 100.0,
        kappa_ems: float = 100.0,
        window_size: int = 3,
        n_em_iters: int = 5,
        learn_kappa: bool = False,
    ):
        self.K = num_classes
        self.D = feature_dim
        self.device = device
        self.window_size = window_size
        self.n_em_iters = n_em_iters
        self.learn_kappa = learn_kappa

        # Concentration parameters (scalar, shared across classes)
        self.kappa_trans = torch.tensor(kappa_trans, dtype=torch.float32, device=device)
        self.kappa_ems = torch.tensor(kappa_ems, dtype=torch.float32, device=device)

        # Prior from source model – normalised columns
        source_prototypes = source_prototypes.to(device).float()
        self.mu0 = F.normalize(source_prototypes, dim=0)  # (D, K)
        # kappa0 must be large relative to D so A_D(kappa0) ≈ 1,
        # ensuring source prototypes are preserved at init.
        self.kappa0 = torch.full((self.K,), max(100.0, 10.0 * feature_dim), device=device)

        # Current posterior: rho_{t,k} (mean direction), gamma_{t,k} (concentration)
        self.rho = self.mu0.clone()  # (D, K)
        self.gamma = self.kappa0.clone()  # (K,)
        self._is_first_batch = True

        # Mixing coefficients – uniform init
        self.pi = torch.full((self.K,), 1.0 / self.K, device=device)

        # Sliding window buffer: list of (H_tau, rho_tau, gamma_tau)
        self.history = []

    # ── E-step helpers ───────────────────────────────────────────

    def _compute_assignments(self, H):
        """Soft cluster assignments  lambda_{t,n,k}.

        H: (N, D)  normalised features
        Returns: (N, K) soft assignments (sum-to-1 over k for each n)
        """
        # E_q[w_{t,k}] = A_D(gamma_k) * rho_k
        A = _log_bessel_ratio(self.D, self.gamma)  # (K,)
        Eq_w = A.unsqueeze(0) * self.rho             # (D, K)

        # log lambda_{t,n,k} = log pi_k + kappa_ems * Eq[w_k]^T h_n  (+ const)
        log_assign = torch.log(self.pi.clamp(min=1e-30)) + self.kappa_ems * (H @ Eq_w)  # (N, K)
        # Softmax for numerical stability
        assignments = torch.softmax(log_assign, dim=1)
        return assignments

    def _update_prototypes(self, H, assignments, prev_rho, prev_gamma, is_first=False):
        """Compute new posterior rho, gamma from E-step quantities.

        Eq (36-37) from the paper:
          beta_{t,k} = kappa_0 * mu_0  (t=1, source prior)
                     + kappa_trans * E_q[w_{t-1,k}]  (t>1)
                     + kappa_ems * sum_n lambda_{n,k} * h_n
          rho_{t,k}  = beta / ||beta||
          gamma_{t,k} = ||beta||
        """
        if is_first:
            # At t=1: use source prior directly (no A_D scaling)
            # Paper Eq. 37 boundary term: kappa_0 * mu_0
            prior_term = self.kappa0.unsqueeze(0) * self.mu0  # (D, K)
        else:
            # At t>1: use previous posterior's expectation
            A_prev = _log_bessel_ratio(self.D, prev_gamma)  # (K,)
            Eq_prev = A_prev.unsqueeze(0) * prev_rho        # (D, K)
            prior_term = self.kappa_trans * Eq_prev          # (D, K)

        # Data contribution: kappa_ems * sum_n lambda_{n,k} * h_n
        # assignments: (N, K), H: (N, D) → weighted sum: (D, K)
        data_term = self.kappa_ems * (H.t() @ assignments)  # (D, K)

        beta = prior_term + data_term  # (D, K)
        gamma_new = beta.norm(dim=0)   # (K,)
        rho_new = F.normalize(beta, dim=0)  # (D, K)
        return rho_new, gamma_new

    # ── M-step ──────────────────────────────────────────────────

    def _update_pi(self, all_assignments):
        """Update mixing coefficients from ALL assignments in the window.

        Uses a Dirichlet prior (alpha=1 per class = uniform) to prevent
        collapse when batch sizes are small.
        """
        # all_assignments: list of (N_t, K) tensors over the window
        total = torch.zeros(self.K, device=self.device)
        count = 0
        for lam in all_assignments:
            total += lam.sum(dim=0)
            count += lam.size(0)
        # Add Dirichlet smoothing (alpha=1 per class)
        smoothed = total + 1.0
        self.pi = smoothed / smoothed.sum()

    def _update_kappa(self, H_list, rho_list, gamma_list, assign_list):
        """MLE update for kappa_trans and kappa_ems (optional)."""
        if len(rho_list) < 2:
            return

        # kappa_trans: r_bar from consecutive prototypes
        numerator = torch.zeros(1, device=self.device)
        count = 0
        for t in range(1, len(rho_list)):
            A_prev = _log_bessel_ratio(self.D, gamma_list[t - 1])
            Eq_prev = A_prev.unsqueeze(0) * rho_list[t - 1]
            A_curr = _log_bessel_ratio(self.D, gamma_list[t])
            Eq_curr = A_curr.unsqueeze(0) * rho_list[t]
            numerator += (Eq_prev * Eq_curr).sum()
            count += self.K
        if count > 0:
            r_bar_trans = (numerator / count).clamp(min=1e-6, max=1.0 - 1e-6)
            self.kappa_trans = _estimate_kappa(r_bar_trans, self.D).clamp(min=1.0)

        # kappa_ems: r_bar from assignments-weighted prototypes vs features
        num_ems = torch.zeros(1, device=self.device)
        denom_ems = 0.0
        for t in range(len(H_list)):
            A_t = _log_bessel_ratio(self.D, gamma_list[t])
            Eq_t = A_t.unsqueeze(0) * rho_list[t]  # (D, K)
            lam = assign_list[t]  # (N_t, K)
            # sum_n sum_k lambda_{n,k} * Eq[w_k]^T h_n
            # = sum_k (Eq[w_k]^T (sum_n lambda_{n,k} h_n))
            weighted_H = H_list[t].t() @ lam  # (D, K)
            num_ems += (Eq_t * weighted_H).sum()
            denom_ems += lam.sum().item()
        if denom_ems > 0:
            r_bar_ems = (num_ems / denom_ems).clamp(min=1e-6, max=1.0 - 1e-6)
            self.kappa_ems = _estimate_kappa(r_bar_ems, self.D).clamp(min=1.0)

    # ── Full EM step ────────────────────────────────────────────

    @torch.no_grad()
    def update(self, H):
        """Run one variational EM cycle given a new batch H (N, D)."""
        H = F.normalize(H.float().to(self.device), dim=1)

        # Save previous posterior for transition prior
        prev_rho = self.rho.clone()
        prev_gamma = self.gamma.clone()

        # --- Sliding window ---
        # We keep the last `window_size` batches
        self.history.append({
            'H': H,
            'prev_rho': prev_rho,
            'prev_gamma': prev_gamma,
        })
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # --- Iterate EM ---
        for _ in range(self.n_em_iters):
            rho_list, gamma_list, assign_list, H_list = [], [], [], []

            for idx, entry in enumerate(self.history):
                H_tau = entry['H']

                # E-step: assignments
                assignments = self._compute_assignments(H_tau)

                # E-step: prototype update
                if idx == 0 and self._is_first_batch:
                    # First batch ever – use source prior directly
                    rho_new, gamma_new = self._update_prototypes(
                        H_tau, assignments, self.mu0, self.kappa0,
                        is_first=True,
                    )
                else:
                    # Use previous time-step's posterior as prior
                    p_rho = rho_list[-1] if idx > 0 else entry['prev_rho']
                    p_gamma = gamma_list[-1] if idx > 0 else entry['prev_gamma']
                    rho_new, gamma_new = self._update_prototypes(
                        H_tau, assignments, p_rho, p_gamma,
                        is_first=False,
                    )

                rho_list.append(rho_new)
                gamma_list.append(gamma_new)
                assign_list.append(assignments)
                H_list.append(H_tau)

            # M-step: pi (over entire window)
            self._update_pi(assign_list)

            # M-step: kappa (optional)
            if self.learn_kappa:
                self._update_kappa(H_list, rho_list, gamma_list, assign_list)

            # Set current posterior to last time-step in window
            self.rho = rho_list[-1]
            self.gamma = gamma_list[-1]

        self._is_first_batch = False

    def predict(self, H):
        """Return logit scores for features H (N, D)."""
        H = F.normalize(H.float().to(self.device), dim=1)
        # Classification: softmax(W_t^T h) where W_t = rho
        logits = H @ self.rho  # (N, K) – cosine similarities
        return logits


# ── Evaluation loop ──────────────────────────────────────────────────

def run_stad(
    loader,
    clip_model,
    clip_weights,
    device,
    kappa_trans=100.0,
    kappa_ems=100.0,
    window_size=3,
    n_em_iters=5,
    batch_size=64,
    learn_kappa=False,
    max_samples=None,
    return_details=False,
):
    """Evaluate STAD-vMF on a standard test stream.

    Accumulates `batch_size` samples before running one EM update,
    then generates predictions with the updated prototypes.

    Returns either scalar accuracy or a detail dict matching TDA's
    return_details format.
    """
    K = clip_weights.size(1)
    D = clip_weights.size(0)

    model = STADvMF(
        num_classes=K,
        feature_dim=D,
        source_prototypes=clip_weights,
        device=device,
        kappa_trans=kappa_trans,
        kappa_ems=kappa_ems,
        window_size=window_size,
        n_em_iters=n_em_iters,
        learn_kappa=learn_kappa,
    )

    accuracies = []
    correct_history = []
    step_times = []
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    run_start = time.perf_counter()

    # Collect samples into batches
    batch_features = []
    batch_targets = []
    sample_idx = 0

    for images, target in tqdm(loader, desc='STAD processed'):
        if max_samples is not None and sample_idx >= max_samples:
            break

        t0 = time.perf_counter()
        with torch.no_grad():
            images = images.to(device)
            feats = clip_model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.float()

        batch_features.append(feats)
        batch_targets.append(target.to(device))
        sample_idx += 1

        # When we have a full batch, run EM + predict
        if len(batch_features) >= batch_size or (max_samples is not None and sample_idx >= max_samples):
            H = torch.cat(batch_features, dim=0)  # (B, D)
            targets = torch.cat(batch_targets, dim=0)

            # Update prototypes with this batch
            model.update(H)

            # Predict on this batch
            logits = model.predict(H)

            # One sample at a time accuracy for consistency
            for j in range(logits.size(0)):
                pred_label = logits[j].argmax().item()
                gt = targets[j].item()
                is_correct = float(pred_label == gt)
                correct_history.append(is_correct)
                accuracies.append(100.0 * is_correct)

            step_times.append(time.perf_counter() - t0)
            batch_features = []
            batch_targets = []

    # Process remaining partial batch
    if batch_features:
        H = torch.cat(batch_features, dim=0)
        targets = torch.cat(batch_targets, dim=0)
        model.update(H)
        logits = model.predict(H)
        for j in range(logits.size(0)):
            pred_label = logits[j].argmax().item()
            gt = targets[j].item()
            is_correct = float(pred_label == gt)
            correct_history.append(is_correct)
            accuracies.append(100.0 * is_correct)

    total_runtime = time.perf_counter() - run_start
    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0

    print(f"---- STAD-vMF accuracy: {avg_acc:.2f}% ----")

    if not return_details:
        return avg_acc

    # Compute forgetting metrics
    cumulative_accuracy = []
    forgetting_curve = []
    running_best = 0.0
    correct_sum = 0.0
    for idx, c in enumerate(correct_history, 1):
        correct_sum += c
        cur = 100.0 * correct_sum / idx
        cumulative_accuracy.append(cur)
        running_best = max(running_best, cur)
        forgetting_curve.append(running_best - cur)

    peak_mem = 0.0
    peak_key = 'peak_device_memory_mb'
    if device.type == 'cuda' and torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_key = 'peak_cuda_memory_mb'
    elif device.type == 'mps' and hasattr(torch.mps, 'current_allocated_memory'):
        peak_mem = torch.mps.current_allocated_memory() / (1024 * 1024)
        peak_key = 'peak_mps_memory_mb'

    return {
        'accuracy': avg_acc,
        'num_samples': len(accuracies),
        'cumulative_accuracy': cumulative_accuracy,
        'forgetting_curve': forgetting_curve,
        'final_forgetting': forgetting_curve[-1] if forgetting_curve else 0.0,
        'mean_forgetting': sum(forgetting_curve) / len(forgetting_curve) if forgetting_curve else 0.0,
        'avg_step_time_ms': (sum(step_times) / len(step_times) * 1000.0) if step_times else 0.0,
        'total_runtime_s': total_runtime,
        peak_key: peak_mem,
        'adapter_memory_mb': 0.0,
    }


# ── CLI ──────────────────────────────────────────────────────────────

def get_arguments():
    parser = argparse.ArgumentParser(description='STAD-vMF baseline evaluation')
    parser.add_argument('--config', required=True, help='Path to config dir')
    parser.add_argument('--datasets', required=True, help='Dataset names separated by /')
    parser.add_argument('--data-root', default='./dataset/')
    parser.add_argument('--backbone', default='RN50', choices=['RN50', 'ViT-B/16'])
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--kappa-trans', type=float, default=1000.0)
    parser.add_argument('--kappa-ems', type=float, default=1000.0)
    parser.add_argument('--window-size', type=int, default=3)
    parser.add_argument('--n-em-iters', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learn-kappa', action='store_true')
    parser.add_argument('--max-samples', type=int, default=None)
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

    for dataset_name in args.datasets.split('/'):
        print(f"\n=== STAD-vMF on {dataset_name} ===")
        cfg = get_config_file(args.config, dataset_name)
        test_loader, classnames, template = build_test_data_loader(
            dataset_name, args.data_root, preprocess
        )
        clip_weights = clip_classifier(classnames, template, clip_model, device=device)

        result = run_stad(
            loader=test_loader,
            clip_model=clip_model,
            clip_weights=clip_weights,
            device=device,
            kappa_trans=args.kappa_trans,
            kappa_ems=args.kappa_ems,
            window_size=args.window_size,
            n_em_iters=args.n_em_iters,
            batch_size=args.batch_size,
            learn_kappa=args.learn_kappa,
            max_samples=args.max_samples,
            return_details=True,
        )

        print(f"  Accuracy : {result['accuracy']:.2f}%")
        print(f"  Samples  : {result['num_samples']}")
        print(f"  Runtime  : {result['total_runtime_s']:.1f}s")
        print(f"  Forgetting (final): {result['final_forgetting']:.3f}")
        print(f"  Forgetting (mean) : {result['mean_forgetting']:.3f}")


if __name__ == '__main__':
    main()
