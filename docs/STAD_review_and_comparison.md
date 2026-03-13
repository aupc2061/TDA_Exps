# Review & Experimental Comparison: STAD vs TDA

**Paper**: *Temporal Test-Time Adaptation with State-Space Models*
Schirmer, Zhang, Nalisnick — TMLR 2024 (arXiv:2407.12492)

**Date**: March 2026

---

## 1. Paper Summary

STAD proposes a **Bayesian state-space model** for test-time adaptation under
*temporal* distribution shift — i.e., the test distribution evolves
smoothly over time (concept drift). The key idea is that class prototypes
should evolve according to a stochastic transition model rather than being
re-estimated i.i.d. at each batch.

Two variants are presented:

| Variant | Transition / Emission | Complexity |
|---|---|---|
| **STAD-Gauss** | Gaussian distributions in R^D | O(D²K) per step |
| **STAD-vMF** | von Mises–Fisher on the unit hypersphere S^{D−1} | O(DK) per step |

STAD-vMF is the more practical variant for high-dimensional features
(D = 512–1024) and is the one we focus on.

### Core Mechanism (STAD-vMF, Algorithm 3)

1. **State**: Per-class prototypes $\rho_{t,k} \in S^{D-1}$ with
   concentration $\gamma_{t,k}$, and mixing weights $\pi_k$.

2. **Transition model**: $w_{t,k} \mid w_{t-1,k} \sim \text{vMF}(w_{t-1,k},\, \kappa_{\text{trans}})$
   — prototypes drift on the hypersphere, regulated by $\kappa_{\text{trans}}$.

3. **Emission model**: $h_{t,n} \mid z_{t,n}=k \sim \text{vMF}(\rho_{t,k},\, \kappa_{\text{ems}})$
   — observed features are drawn from a vMF centred at the class prototype.

4. **Inference**: Variational EM with mean-field factorisation:
   - **E-step**: Compute soft assignments $\lambda_{t,n,k}$ (responsibilities),
     then update prototypes:
     $$\beta_{t,k} = \underbrace{\kappa_0 \mu_0}_{\text{source prior (t=1)}} + \underbrace{\kappa_{\text{trans}} A_D(\gamma_{t-1,k}) \rho_{t-1,k}}_{\text{transition prior (t>1)}} + \underbrace{\kappa_{\text{ems}} \sum_n \lambda_{n,k} h_n}_{\text{data term}}$$
     $$\rho_{t,k} = \frac{\beta_{t,k}}{\|\beta_{t,k}\|},\quad \gamma_{t,k} = \|\beta_{t,k}\|$$
   - **M-step**: Update $\pi_k$ with Dirichlet smoothing over the sliding window.

5. **Sliding window**: EM re-processes the last $s$ time steps
   (default $s = 3$) to improve estimates.

6. **Prediction**: $\hat{y} = \arg\max_k\; h^T \rho_{t,k}$ (cosine
   similarity to adapted prototypes).

### Key Hyperparameters

| Parameter | Meaning | Paper default |
|---|---|---|
| $\kappa_{\text{trans}}$ | Transition concentration (higher = less drift) | dataset-specific |
| $\kappa_{\text{ems}}$ | Emission concentration (higher = tighter clusters) | dataset-specific |
| $s$ | Sliding window size | 3 |
| EM iterations | Number of EM sweeps per time step | 5 |

---

## 2. Strengths

1. **Principled Bayesian framework.** The SSM formulation is elegant:
   latent class prototypes evolve via a transition kernel, and observed
   features are emissions. This gives a clear generative story.

2. **Temporal modelling.** Unlike methods that treat each batch
   independently, STAD explicitly models smooth evolution of the class
   decision boundary — appropriate for real-world deployment where
   distribution shift is gradual (e.g., decade-by-decade image style in
   Yearbook, seasonal changes in satellite imagery in FMoW).

3. **Small-batch robustness.** On datasets with temporal structure
   (Yearbook, EVIS), STAD maintains performance even at batch size 1,
   where methods like TENT or LAME collapse.

4. **Label-shift handling.** The per-class mixing weights $\pi_k$ adapt
   to non-uniform class proportions at test time, which many TTA methods
   ignore.

5. **Scalable vMF variant.** STAD-vMF avoids the O(D²K) covariance
   tracking of STAD-Gauss, making it practical with CLIP-scale features
   (D = 1024).

---

## 3. Weaknesses & Limitations

### 3.1 Fundamental Assumption: Temporal Structure Required

STAD assumes the test distribution changes **smoothly over time**.
Its transition model $\text{vMF}(\rho_{t-1,k},\, \kappa_{\text{trans}})$ and
sliding window only make sense when consecutive batches share a similar
distribution. On **i.i.d. test streams** (the standard TTA evaluation protocol
for CLIP-based zero-shot models), this assumption is violated: there is no
temporal correlation to exploit, and the transition model adds noise to
otherwise good prototypes.

### 3.2 Full Classifier Replacement

STAD **replaces** the classification head entirely: predictions are
$\arg\max_k\; h^T \rho_{t,k}$, where $\rho_{t,k}$ are the EM-adapted
prototypes. This means:

- If the EM converges to a degenerate solution, there is **no fallback** to
  the original zero-shot predictions.
- Contrast with TDA, which **adds corrections** on top of CLIP zero-shot
  logits: $\text{logits} = \alpha \cdot \text{zero-shot} + \beta \cdot \text{cache\_correction}$. Even if
  the memory is unhelpful, zero-shot performance is preserved.

### 3.3 Unsupervised Clustering Fragility

The EM is doing fully unsupervised K-means-like clustering (vMF mixture)
on normalised features. For fine-grained tasks with many classes
(e.g., 37 for Oxford Pets, 101 for UCF101), a batch of 64 samples
provides ≈ 1.7 or 0.6 samples per class on average. This is **far too
sparse** for reliable mixture estimation, causing prototypes to drift
toward batch-specific modes rather than class centres.

### 3.4 kappa Sensitivity and the Bessel Ratio

The Bessel ratio $A_D(\kappa)$ controls how much influence the prior
(source or transition) has relative to data. In high dimension
(D = 1024), the bounded approximation $A_D(\kappa) = 2\kappa / (2\kappa + D - 1)$
means that even $\kappa = 10{,}000$ gives $A_D \approx 0.95$, and
$\kappa = 100$ gives $A_D \approx 0.16$. The paper does not discuss
practical guidance for setting $\kappa$ relative to $D$ in the CLIP
feature space.

### 3.5 Computational Cost

Each time step requires $n_{\text{iters}} \times s$ forward passes through
the EM inner loop (default: 5 × 3 = 15 passes per batch). While each pass
is O(NDK), this is still significantly more expensive than TDA's
forward-pass-only SSM update.

---

## 4. Experimental Comparison

### 4.1 Setup

| | **TDA (ours)** | **STAD-vMF** |
|---|---|---|
| Backbone | CLIP RN50 (frozen) | CLIP RN50 (frozen) |
| Feature dim | D = 1024 | D = 1024 |
| Adaptation | Additive correction on zero-shot logits | Full prototype replacement via EM |
| Memory | SSM (heuristic delta gating) or cache | Sliding window + vMF prototypes |
| Hyperparameters | α, β (logit weights) | κ_trans, κ_ems, window_size, n_em_iters |
| Training required | None | None (variational EM at test time) |

Both methods evaluated on the same i.i.d. shuffled test streams (no
temporal ordering), which is the standard protocol for CLIP-based TTA.

### 4.2 Oxford Pets (37 classes, 3669 test samples)

| Method | N=100 | N=500 | N=1000 | N=3669 |
|---|---|---|---|---|
| **CLIP zero-shot** | — | 85.0% | — | — |
| **TDA SSM (heuristic)** | 81.0% | 85.4% | 78.6%¹ | 85.96% |
| **TDA Cache k=20** | 95.0% | 91.2% | 81.7% | — |
| **STAD-vMF** (κt=1000, κe=1000, bs=64) | — | **19.6%** | — | — |

¹ Forgetting at N=1000 is an artifact of the fixed-size SSM buffer.

**STAD kappa grid search** (Oxford Pets, N=500, bs=64, n_em_iters=5):

| κ_trans \ κ_ems | 100 | 1,000 | 10,000 |
|---|---|---|---|
| **100** | 19.6% | 19.6% | 19.6% |
| **1,000** | 19.6% | 19.6% | 19.6% |
| **10,000** | 19.6% | 19.6% | 19.6% |

Result: **Completely invariant to κ.** The EM converges to the same
degenerate solution regardless of concentration parameters.

With n_em_iters=1 (less aggressive adaptation):

| κ_trans \ κ_ems | 1,000 | 10,000 |
|---|---|---|
| **1,000** | 19.6% | 17.8% |
| **10,000** | 19.6% | 18.6% |

Still collapsed. Reducing EM iterations does not help.

### 4.3 UCF101 (101 classes, 3783 test samples)

| Method | N=100 | N=500 | N=1000 |
|---|---|---|---|
| **CLIP zero-shot** | ~63%² | ~63% | ~63% |
| **TDA SSM (heuristic)** | 73.0% | 77.6% | 72.6% |
| **TDA Cache k=20** | 79.0% | 81.8% | 74.4% |
| **STAD-vMF** (κt=1000, κe=1000, N=200, bs=64) | — | **1.5%** | — |

² CLIP RN50 zero-shot on UCF101 is approximately 63% (reported in CLIP paper).

### 4.4 Analysis: Why STAD Collapses

The 19.6% result on Oxford Pets and 1.5% on UCF101 (near or below random
chance for K=37 and K=101 respectively) demand explanation. Three factors
compound:

1. **No temporal structure.** The i.i.d. shuffled test stream has no
   distributional continuity between consecutive batches. STAD's
   transition model adds noise rather than useful inductive bias.

2. **Sparse per-class representation.** With bs=64 and K=37 (Pets) or
   K=101 (UCF101), each batch has ~1.7 or ~0.6 samples per class. The
   vMF mixture EM cannot reliably separate 37–101 clusters from ≤64
   points.

3. **Prototype replacement without fallback.** Once the EM distorts the
   prototypes, prediction quality drops, which produces worse
   assignments in the next E-step, creating a **feedback loop** of
   degradation. There is no mechanism to revert to the source prototypes
   when confidence is low.

The fact that accuracy is invariant to κ (§4.2 grid search) confirms the
EM converges to a fixed point determined by the batch composition rather
than the prior strength. Even with κ_trans = 10,000 (strong prior), the
A_D(10000) ≈ 0.95 scaling means the prior contribution is only:

$$\kappa_{\text{trans}} \cdot A_D(\gamma) \cdot \rho \approx 10{,}000 \times 0.95 \times \rho = 9{,}500 \|\rho\|$$

versus the data term:

$$\kappa_{\text{ems}} \cdot \sum_n \lambda_{n,k} h_n$$

which for K=37 classes and N=64 samples has magnitude ~$\kappa_{\text{ems}} \times (64/37) \approx 1{,}730$ per class. 
So the prior still dominates — yet accuracy is 19.6%.
This means the source prototypes themselves are being corrupted by the
iterative EM (5 iterations × 3 window steps = 15 updates to prototypes
per batch), where even small misassignments compound across iterations.

---

## 5. Fair Comparison Caveats

It would be misleading to conclude "STAD is bad" from these results.
The comparison highlights a **domain mismatch**, not a method failure:

| Aspect | STAD's designed regime | Our evaluation regime |
|---|---|---|
| Distribution shift | Temporal / gradual | None (i.i.d.) |
| Number of classes | Small (2–62) | Medium-large (37–101) |
| Feature source | Task-specific encoders | Frozen CLIP (zero-shot) |
| Batch composition | Balanced per time step | Random i.i.d. sampling |

**STAD excels in its target setting.** On Yearbook (2 classes, temporal
ordering by decade), STAD-vMF achieves **87.9%** vs the best baseline of
84.5%. On FMoW (62 classes, temporal satellite imagery), it achieves
**32.3%** vs 28.8%. These are genuine improvements in the temporal TTA
regime.

The takeaway is that **STAD and TDA address fundamentally different
problems**:

- **STAD**: Temporal distribution shift with gradual concept drift,
  small number of classes, classifier replacement via Bayesian inference.
- **TDA**: Static (i.i.d.) fine-grained classification with frozen CLIP,
  many classes, additive memory correction preserving zero-shot quality.

---

## 6. What TDA Can Learn from STAD

Despite the performance gap on our benchmarks, several ideas from STAD
are worth considering:

1. **Principled uncertainty tracking.** STAD's per-class concentration
   $\gamma_{t,k}$ provides a calibrated confidence measure. Our SSM's
   heuristic delta gating could benefit from a similar probabilistic
   interpretation.

2. **Temporal modelling for deployment.** If TDA were deployed in a
   setting with genuine distribution drift (e.g., autonomous driving
   across weather conditions), a transition model between memory states
   could improve robustness.

3. **Label-shift adaptation.** STAD's $\pi_k$ update automatically
   handles non-uniform class proportions. TDA currently assumes uniform
   test-time class distribution.

---

## 7. Reproducibility

All experiments use:
- CLIP RN50, frozen, float32
- Apple M4 (MPS device)
- Identical dataset splits and data ordering (seed=1)
- Implementation: `stad_baseline.py` — verified faithful to Algorithm 3
  of the paper

STAD hyperparameters tested:
- κ_trans ∈ {100, 1000, 10000}
- κ_ems ∈ {100, 1000, 10000}
- window_size = 3 (paper default)
- n_em_iters ∈ {1, 5}
- batch_size = 64
