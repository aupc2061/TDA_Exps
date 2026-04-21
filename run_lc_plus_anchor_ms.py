#!/usr/bin/env python3
"""
LC+ Anchor-Mamba runner.

Standalone LC+ variant that avoids the missing m3_cache_memory.py dependency,
keeps the LC+ lazy-commit/prototype-correction pipeline, and optionally adds
the ViT anchor reservoir from the main TDA experiments.
"""

import argparse
import csv
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import clip
from clip.model import VisionTransformer
from memory import AnchorReservoir, CacheMemory
from utils import (
    build_test_data_loader,
    clip_classifier,
    cls_acc,
    get_clip_logits,
    get_clip_logits_with_details,
    get_config_file,
    get_entropy,
    resolve_device,
)


class LCPlusMamba3CacheMemory:
    """Minimal cache + Mamba3-style state memory used by LC+.

    The cache is used for prototype construction. The Mamba-style state is used
    as an additional low-rank class memory during logit correction.
    """

    def __init__(
        self,
        num_classes,
        feature_dim,
        device,
        shot_capacity,
        mode="mamba3-trapezoid",
        decay=0.96,
        min_blend=0.02,
        max_blend=0.35,
        entropy_temp=4.0,
        phase_strength=0.15,
        state_logit_weight=0.5,
    ):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.shot_capacity = shot_capacity
        self.mode = mode
        self.decay = decay
        self.min_blend = min_blend
        self.max_blend = max_blend
        self.entropy_temp = entropy_temp
        self.phase_strength = phase_strength if mode == "mamba3-complex" else 0.0
        self.state_logit_weight = state_logit_weight
        self.eps = 1e-8

        self.cache = {}
        self.state_keys = torch.zeros(num_classes, feature_dim, dtype=torch.float32, device=device)
        self.prev_inputs = torch.zeros(num_classes, feature_dim, dtype=torch.float32, device=device)
        self.seen = torch.zeros(num_classes, dtype=torch.bool, device=device)
        self.total_updates_seen = 0
        self.accepted_updates = 0

    def is_empty(self):
        return len(self.cache) == 0 and not bool(torch.any(self.seen).item())

    def _phase_rotate(self, vector, phase):
        if abs(phase) <= self.eps:
            return vector
        rotated = vector.clone()
        n = (vector.numel() // 2) * 2
        if n == 0:
            return rotated
        c = math.cos(phase)
        s = math.sin(phase)
        rotated[:n:2] = c * vector[:n:2] - s * vector[1:n:2]
        rotated[1:n:2] = s * vector[:n:2] + c * vector[1:n:2]
        return rotated

    def _update_cache(self, pred, feature, loss):
        item = [feature.detach(), float(loss)]
        if pred not in self.cache:
            self.cache[pred] = [item]
            return
        class_items = self.cache[pred]
        if len(class_items) < self.shot_capacity:
            class_items.append(item)
        elif float(loss) < class_items[-1][1]:
            class_items[-1] = item
        class_items.sort(key=lambda x: x[1])

    def _update_state(self, pred, feature, normalized_entropy):
        obs = F.normalize(feature.squeeze(0).float(), dim=-1)
        if not self.seen[pred]:
            self.state_keys[pred] = obs
            self.prev_inputs[pred] = obs
            self.seen[pred] = True
            return

        prev = self.state_keys[pred]
        prev_input = self.prev_inputs[pred]
        sim = F.cosine_similarity(prev.unsqueeze(0), obs.unsqueeze(0)).item()
        confidence = max(0.0, min(1.0, 1.0 - float(normalized_entropy)))
        novelty = max(0.0, min(1.0, 1.0 - sim))
        base_gate = torch.sigmoid(torch.tensor(confidence * self.entropy_temp, device=self.device)).item()
        delta_t = self.min_blend + (self.max_blend - self.min_blend) * min(1.0, base_gate * novelty)
        alpha_t = math.exp(-delta_t * (1.0 - self.decay))
        gamma_t = delta_t * (0.5 + 0.5 * confidence)
        beta_t = delta_t * (1.0 - (0.5 + 0.5 * confidence)) * alpha_t
        phase = self.phase_strength * novelty * (2.0 * confidence - 1.0)

        if self.mode == "mamba3-complex":
            prev = self._phase_rotate(prev, phase)
            prev_input = self._phase_rotate(prev_input, phase)
            obs = self._phase_rotate(obs, phase)

        updated = alpha_t * prev + beta_t * prev_input + gamma_t * obs
        self.state_keys[pred] = F.normalize(updated, dim=-1)
        self.prev_inputs[pred] = obs

    def update(self, pred, feature, loss, prop_entropy=0.0, clip_weights=None):
        with torch.no_grad():
            self.total_updates_seen += 1
            pred = int(pred)
            self._update_cache(pred, feature, loss)
            self._update_state(pred, feature, prop_entropy)
            self.accepted_updates += 1
            return True

    def _cache_logits(self, image_features, alpha, beta, clip_weights):
        if len(self.cache) == 0:
            return torch.zeros_like(image_features @ clip_weights)
        keys, values = [], []
        for class_index in sorted(self.cache.keys()):
            for item in self.cache[class_index]:
                keys.append(item[0])
                values.append(class_index)
        keys = torch.cat(keys, dim=0).to(image_features.device).permute(1, 0)
        values = F.one_hot(
            torch.tensor(values, dtype=torch.int64, device=image_features.device),
            num_classes=clip_weights.size(1),
        ).to(image_features.dtype)
        affinity = image_features @ keys
        return alpha * (((-1.0) * (beta - beta * affinity)).exp() @ values)

    def _state_logits(self, image_features, alpha, beta, clip_weights):
        if not bool(torch.any(self.seen).item()):
            return torch.zeros_like(image_features @ clip_weights)
        active_idx = torch.where(self.seen)[0]
        keys = F.normalize(self.state_keys[active_idx], dim=-1).permute(1, 0)
        values = F.one_hot(active_idx, num_classes=clip_weights.size(1)).to(image_features.dtype)
        affinity = image_features.float() @ keys
        return alpha * (((-1.0) * (beta - beta * affinity)).exp() @ values.float()).to(image_features.dtype)

    def logits(self, image_features, alpha, beta, clip_weights):
        cache_logits = self._cache_logits(image_features, alpha, beta, clip_weights)
        state_logits = self._state_logits(image_features, alpha, beta, clip_weights)
        return cache_logits + self.state_logit_weight * state_logits

    def stats(self):
        cache_entries = sum(len(items) for items in self.cache.values())
        active_classes = int(torch.sum(self.seen).item())
        return {
            "accepted_updates": int(self.accepted_updates),
            "total_update_attempts": int(self.total_updates_seen),
            "accept_rate": float(self.accepted_updates / self.total_updates_seen) if self.total_updates_seen else 0.0,
            "cache_entries": int(cache_entries),
            "active_state_classes": active_classes,
        }

    def memory_bytes(self):
        total = self.state_keys.element_size() * self.state_keys.nelement()
        total += self.prev_inputs.element_size() * self.prev_inputs.nelement()
        total += self.seen.element_size() * self.seen.nelement()
        for items in self.cache.values():
            for item in items:
                total += item[0].element_size() * item[0].nelement()
        return total


class MultiStateSSM:
    def __init__(self, num_classes, feature_dim, max_states=3, spawn_threshold=0.5, decay=0.96, device="cuda"):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.max_states = max_states
        self.spawn_threshold = spawn_threshold
        self.decay = decay
        self.device = device
        self.states = [[] for _ in range(num_classes)]
        self.total_updates = 0
        self.eps = 1e-8

    def update(self, pred, feature, normalized_entropy):
        pred = int(pred)
        obs = F.normalize(feature.squeeze(0).float(), dim=-1)
        states = self.states[pred]
        if not states:
            states.append({"eta": obs.clone(), "count": 1})
            self.total_updates += 1
            return
        sims = []
        for state in states:
            eta = state["eta"]
            sims.append(float(F.cosine_similarity(F.normalize(eta, dim=-1).unsqueeze(0), obs.unsqueeze(0)).item()))
        best_idx = int(np.argmax(sims))
        if sims[best_idx] < self.spawn_threshold and len(states) < self.max_states:
            states.append({"eta": obs.clone(), "count": 1})
        else:
            confidence = max(0.0, min(1.0, 1.0 - float(normalized_entropy)))
            blend = 0.05 + 0.30 * confidence
            states[best_idx]["eta"] = F.normalize((1.0 - blend) * states[best_idx]["eta"] + blend * obs, dim=-1)
            states[best_idx]["count"] += 1
        self.total_updates += 1

    def get_all_protos(self, class_idx):
        out = []
        for state in self.states[class_idx]:
            kappa = float(state["count"])
            out.append((F.normalize(state["eta"], dim=-1), kappa))
        return out

    def stats(self):
        return {
            "ms_total_states": sum(len(states) for states in self.states),
            "ms_active_classes": sum(1 for states in self.states if states),
            "ms_updates": self.total_updates,
        }


def tensor_entropy(logits):
    probs = logits.softmax(dim=-1)
    return float((-(probs * torch.log(probs + 1e-8)).sum()).item())


def tensor_confidence(logits):
    return float(logits.softmax(dim=-1).max().item())


def precompute_features(loader, clip_model, clip_weights, device, enable_anchor=False, max_samples=None):
    data = []
    clip_model.eval()
    with torch.no_grad():
        for idx, (images, target) in enumerate(tqdm(loader, desc="Pre-encoding")):
            if max_samples is not None and idx >= max_samples:
                break
            target = target.to(device)
            if enable_anchor and isinstance(getattr(clip_model, "visual", None), VisionTransformer):
                image_features, clip_logits, loss, prob_map, pred, details = get_clip_logits_with_details(
                    images,
                    clip_model,
                    clip_weights,
                    device=device,
                    return_layer_cls=True,
                )
                layer_cls_tokens = details.get("layer_cls_tokens")
                supports_anchor = bool(details.get("supports_anchor", False))
            else:
                image_features, clip_logits, loss, prob_map, pred = get_clip_logits(
                    images,
                    clip_model,
                    clip_weights,
                    device=device,
                )
                layer_cls_tokens = None
                supports_anchor = False
            data.append({
                "image_features": image_features,
                "clip_logits": clip_logits,
                "loss": float(loss),
                "prob_map": prob_map,
                "pred": pred,
                "prop_entropy": get_entropy(loss, clip_weights),
                "target": target,
                "layer_cls_tokens": layer_cls_tokens,
                "supports_anchor": supports_anchor,
            })
    return data


def score_with_state(
    image_features,
    clip_logits,
    clip_weights,
    pos_memory,
    neg_memory,
    pos_params,
    neg_params,
    prop_entropy=0.0,
    anchor_memory=None,
    layer_cls_tokens=None,
    anchor_alpha=0.4,
    anchor_beta=1.5,
    anchor_norm_accumulator=None,
):
    final_logits = clip_logits.clone()
    if pos_memory is not None and not pos_memory.is_empty():
        final_logits += pos_memory.logits(image_features, pos_params["alpha"], pos_params["beta"], clip_weights)
    if neg_memory is not None and not neg_memory.is_empty():
        final_logits -= neg_memory.logits(
            image_features,
            neg_params["alpha"],
            neg_params["beta"],
            clip_weights,
            (neg_params["mask_threshold"]["lower"], neg_params["mask_threshold"]["upper"]),
        )
    if anchor_memory is not None and not anchor_memory.is_empty() and layer_cls_tokens is not None:
        anchor_logits = anchor_memory.logits(layer_cls_tokens, anchor_alpha, anchor_beta).to(final_logits.dtype)
        if torch.isfinite(anchor_logits).all():
            final_logits += anchor_logits
            if anchor_norm_accumulator is not None:
                anchor_norm_accumulator.append(float(anchor_logits.norm().item()))
    return final_logits


def compute_proto_logits(image_features, pos_memory, proto_temperature=0.1, proto_min_samples=2):
    cache = pos_memory.cache if hasattr(pos_memory, "cache") else {}
    if not cache:
        return None, 0
    query = F.normalize(image_features.squeeze(0).float(), dim=-1)
    proto_logits = torch.zeros(pos_memory.num_classes, device=query.device, dtype=torch.float32)
    proto_count = 0
    for class_idx, items in cache.items():
        if len(items) < proto_min_samples:
            continue
        feats = torch.stack([item[0].squeeze(0).float() for item in items], dim=0)
        weights = torch.tensor([math.exp(-float(item[1])) for item in items], device=feats.device)
        weights = weights / weights.sum().clamp_min(1e-8)
        centroid = F.normalize((feats * weights.unsqueeze(1)).sum(dim=0), dim=-1)
        reliability = min(1.0, len(items) / max(1, pos_memory.shot_capacity))
        proto_logits[int(class_idx)] = torch.dot(query, centroid) * reliability / max(proto_temperature, 1e-6)
        proto_count += 1
    return proto_logits.unsqueeze(0), proto_count


def compute_multistate_proto_logits(image_features, multi_ssm, proto_temperature=0.1, min_kappa=1.0):
    if multi_ssm is None:
        return None, 0
    query = F.normalize(image_features.squeeze(0).float(), dim=-1)
    proto_logits = torch.zeros(multi_ssm.num_classes, device=query.device, dtype=torch.float32)
    count = 0
    for class_idx in range(multi_ssm.num_classes):
        protos = multi_ssm.get_all_protos(class_idx)
        if not protos:
            continue
        total_kappa = sum(kappa for _, kappa in protos)
        if total_kappa < min_kappa:
            continue
        best_sim = max(float(torch.dot(query, proto).item()) for proto, _ in protos)
        reliability = min(1.0, total_kappa / 5.0)
        proto_logits[class_idx] = best_sim * reliability / max(proto_temperature, 1e-6)
        count += 1
    return proto_logits.unsqueeze(0), count


def choose_blend_logits(candidates, policy="agreement-gated", confidence_gate=0.70):
    if policy == "entropy-min":
        return min(candidates, key=lambda item: tensor_entropy(item[1]))

    base_name, base_logits = "p1", dict(candidates)["p1"]
    base_entropy = tensor_entropy(base_logits)
    base_pred = int(base_logits.argmax(dim=1).item())
    best_name, best_logits, best_entropy = base_name, base_logits, base_entropy

    for name, logits in candidates:
        if name == "p1":
            continue
        ent = tensor_entropy(logits)
        pred = int(logits.argmax(dim=1).item())
        conf = tensor_confidence(logits)
        if ent < best_entropy and (pred == base_pred or conf >= confidence_gate):
            best_name, best_logits, best_entropy = name, logits, ent
    return best_name, best_logits


def run_lc_plus_anchor(
    data,
    clip_weights,
    pos_cfg,
    neg_cfg,
    device,
    buffer_size=30,
    accept_entropy_margin=0.01,
    accept_conf_gate=0.0,
    proto_gamma=1.0,
    proto_temperature=0.1,
    proto_min_samples=2,
    use_multi_state=False,
    num_states=3,
    spawn_threshold=0.5,
    blend_policy="agreement-gated",
    blend_confidence_gate=0.70,
    enable_anchor=False,
    anchor_capacity=2,
    anchor_entropy_threshold=0.15,
    anchor_alpha=0.4,
    anchor_beta=1.5,
    mamba_mode="mamba3-trapezoid",
    state_logit_weight=0.5,
):
    pos_params = {k: pos_cfg[k] for k in ["shot_capacity", "alpha", "beta"]}
    neg_params = {k: neg_cfg[k] for k in ["shot_capacity", "alpha", "beta", "entropy_threshold", "mask_threshold"]}
    num_classes = clip_weights.size(1)
    feature_dim = data[0]["image_features"].size(1)

    pos_memory = LCPlusMamba3CacheMemory(
        num_classes=num_classes,
        feature_dim=feature_dim,
        device=device,
        shot_capacity=pos_params["shot_capacity"],
        mode=mamba_mode,
        state_logit_weight=state_logit_weight,
    )
    neg_memory = CacheMemory(shot_capacity=neg_params["shot_capacity"], include_prob_map=True)
    multi_ssm = MultiStateSSM(num_classes, feature_dim, num_states, spawn_threshold, device=device) if use_multi_state else None
    anchor_memory = AnchorReservoir(num_classes, anchor_capacity, device, anchor_entropy_threshold) if enable_anchor else None

    lc_logits_all = [None] * len(data)
    buffer = []
    rescore_accepts = 0
    rescore_flips = 0
    anchor_accepts = []
    anchor_norms = []

    for sample_idx, sample in enumerate(tqdm(data, desc="LC+ phase 1")):
        image_features = sample["image_features"]
        clip_logits = sample["clip_logits"]
        pred = sample["pred"]
        prop_entropy = sample["prop_entropy"]

        pos_memory.update(pred, image_features, sample["loss"], prop_entropy=prop_entropy, clip_weights=clip_weights)
        if neg_params["entropy_threshold"]["lower"] < prop_entropy < neg_params["entropy_threshold"]["upper"]:
            neg_memory.update(pred, image_features, sample["loss"], prob_map=sample["prob_map"])
        if multi_ssm is not None:
            multi_ssm.update(pred, image_features, prop_entropy)
        if anchor_memory is not None:
            accepted = anchor_memory.update(pred, sample.get("layer_cls_tokens"), prop_entropy)
            anchor_accepts.append(1.0 if accepted else 0.0)

        tentative_logits = score_with_state(
            image_features,
            clip_logits,
            clip_weights,
            pos_memory,
            neg_memory,
            pos_params,
            neg_params,
            prop_entropy,
            anchor_memory,
            sample.get("layer_cls_tokens"),
            anchor_alpha,
            anchor_beta,
            anchor_norms,
        )

        buffer.append({
            "idx": sample_idx,
            "features": image_features,
            "clip_logits": clip_logits,
            "prop_entropy": prop_entropy,
            "layer_cls_tokens": sample.get("layer_cls_tokens"),
            "tentative_logits": tentative_logits.detach(),
        })

        if len(buffer) > buffer_size:
            oldest = buffer.pop(0)
            mature_logits = score_with_state(
                oldest["features"],
                oldest["clip_logits"],
                clip_weights,
                pos_memory,
                neg_memory,
                pos_params,
                neg_params,
                oldest["prop_entropy"],
                anchor_memory,
                oldest.get("layer_cls_tokens"),
                anchor_alpha,
                anchor_beta,
                anchor_norms,
            )
            tent_ent = tensor_entropy(oldest["tentative_logits"])
            mat_ent = tensor_entropy(mature_logits)
            tent_pred = int(oldest["tentative_logits"].argmax(1).item())
            mat_pred = int(mature_logits.argmax(1).item())
            mat_conf = tensor_confidence(mature_logits)
            accept = mat_ent < tent_ent - accept_entropy_margin and (tent_pred == mat_pred or mat_conf > accept_conf_gate)
            lc_logits_all[oldest["idx"]] = mature_logits.detach() if accept else oldest["tentative_logits"]
            if accept:
                rescore_accepts += 1
                if tent_pred != mat_pred:
                    rescore_flips += 1

    for item in buffer:
        mature_logits = score_with_state(
            item["features"],
            item["clip_logits"],
            clip_weights,
            pos_memory,
            neg_memory,
            pos_params,
            neg_params,
            item["prop_entropy"],
            anchor_memory,
            item.get("layer_cls_tokens"),
            anchor_alpha,
            anchor_beta,
            anchor_norms,
        )
        tent_ent = tensor_entropy(item["tentative_logits"])
        mat_ent = tensor_entropy(mature_logits)
        tent_pred = int(item["tentative_logits"].argmax(1).item())
        mat_pred = int(mature_logits.argmax(1).item())
        mat_conf = tensor_confidence(mature_logits)
        accept = mat_ent < tent_ent - accept_entropy_margin and (tent_pred == mat_pred or mat_conf > accept_conf_gate)
        lc_logits_all[item["idx"]] = mature_logits.detach() if accept else item["tentative_logits"]
        if accept:
            rescore_accepts += 1
            if tent_pred != mat_pred:
                rescore_flips += 1

    pass1_logits = []
    proto_logits_all = []
    proto_classes = 0
    ms_proto_classes = 0
    for idx, sample in enumerate(tqdm(data, desc="LC+ rescore/proto")):
        p1 = score_with_state(
            sample["image_features"],
            sample["clip_logits"],
            clip_weights,
            pos_memory,
            neg_memory,
            pos_params,
            neg_params,
            sample["prop_entropy"],
            anchor_memory,
            sample.get("layer_cls_tokens"),
            anchor_alpha,
            anchor_beta,
            anchor_norms,
        )
        pass1_logits.append(p1)
        proto_sim, proto_count = compute_proto_logits(sample["image_features"], pos_memory, proto_temperature, proto_min_samples)
        ms_proto_sim, ms_count = compute_multistate_proto_logits(sample["image_features"], multi_ssm, proto_temperature)
        if idx == 0:
            proto_classes = proto_count
            ms_proto_classes = ms_count
        combined = p1
        if proto_sim is not None:
            combined = combined + proto_gamma * proto_sim.to(combined.dtype)
        if ms_proto_sim is not None:
            combined = combined + proto_gamma * 0.5 * ms_proto_sim.to(combined.dtype)
        proto_logits_all.append(combined)

    blend_logits = []
    blend_from_proto = 0
    for i in range(len(data)):
        name, logits = choose_blend_logits(
            [("lc", lc_logits_all[i]), ("p1", pass1_logits[i]), ("proto", proto_logits_all[i])],
            policy=blend_policy,
            confidence_gate=blend_confidence_gate,
        )
        blend_logits.append(logits)
        if name == "proto":
            blend_from_proto += 1

    lc_avg = sum(cls_acc(lc_logits_all[i], data[i]["target"]) for i in range(len(data))) / len(data)
    pass1_avg = sum(cls_acc(pass1_logits[i], data[i]["target"]) for i in range(len(data))) / len(data)
    proto_avg = sum(cls_acc(proto_logits_all[i], data[i]["target"]) for i in range(len(data))) / len(data)
    blend_avg = sum(cls_acc(blend_logits[i], data[i]["target"]) for i in range(len(data))) / len(data)
    results = {
        "lazy_commit": lc_avg,
        "pass1_rescore": pass1_avg,
        "proto_rescore": proto_avg,
        "per_sample_blend": blend_avg,
    }
    method_used = max(results, key=results.get)
    best_avg = results[method_used]

    pos_stats = pos_memory.stats()
    stats = {
        **pos_stats,
        "lc_acc": lc_avg,
        "pass1_acc": pass1_avg,
        "proto_acc": proto_avg,
        "blend_acc": blend_avg,
        "method_used": method_used,
        "proto_classes": proto_classes,
        "blend_from_proto": blend_from_proto,
        "rescore_accepts": rescore_accepts,
        "rescore_flips": rescore_flips,
        "ms_proto_classes": ms_proto_classes,
        "anchor_update_accept_rate": float(np.mean(anchor_accepts)) if anchor_accepts else 0.0,
        "anchor_fill_ratio": anchor_memory.fill_ratio() if anchor_memory is not None else 0.0,
        "avg_anchor_correction_norm": float(np.mean(anchor_norms)) if anchor_norms else 0.0,
    }
    if multi_ssm is not None:
        stats.update(multi_ssm.stats())
    return best_avg, stats


def parse_args():
    parser = argparse.ArgumentParser(description="LC+ Anchor-Mamba standalone runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--backbone", default="ViT-B/16", choices=["RN50", "ViT-B/16"])
    parser.add_argument("--data-root", default="./dataset/")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--feature-cache-dir", default="feature_cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--proto-gamma", type=float, default=1.0)
    parser.add_argument("--proto-temperature", type=float, default=0.1)
    parser.add_argument("--proto-min-samples", type=int, default=2)
    parser.add_argument("--buffer-size", type=int, default=30)
    parser.add_argument("--mamba3-mode", choices=["mamba3-trapezoid", "mamba3-complex"], default="mamba3-trapezoid")
    parser.add_argument("--state-logit-weight", type=float, default=0.5)
    parser.add_argument("--multi-state", action="store_true")
    parser.add_argument("--num-states", type=int, default=3)
    parser.add_argument("--spawn-threshold", type=float, default=0.5)
    parser.add_argument("--blend-policy", choices=["entropy-min", "agreement-gated"], default="agreement-gated")
    parser.add_argument("--blend-confidence-gate", type=float, default=0.70)
    parser.add_argument("--enable-anchor-reservoir", action="store_true")
    parser.add_argument("--anchor-reservoir-size", type=int, default=2)
    parser.add_argument("--anchor-entropy-threshold", type=float, default=0.15)
    parser.add_argument("--anchor-alpha", type=float, default=0.4)
    parser.add_argument("--anchor-beta", type=float, default=1.5)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    clip_model, preprocess = clip.load(args.backbone, device=device)
    if device.type == "mps":
        clip_model.float()
    clip_model.eval()

    os.makedirs("results/lc_plus_anchor", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backbone_tag = args.backbone.replace("/", "")
    csv_path = os.path.join("results/lc_plus_anchor", f"lc_plus_anchor_{backbone_tag}_{timestamp}.csv")
    ds_alias = {"imagenet": "I", "imagenet_a": "A", "imagenet_v": "V", "imagenet_r": "R", "imagenet_s": "S"}

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "dataset", "method", "accuracy", "lc_acc", "pass1_acc", "proto_acc", "blend_acc",
            "method_used", "proto_classes", "blend_from_proto", "anchor_update_accept_rate",
            "anchor_fill_ratio", "avg_anchor_correction_norm", "cache_entries", "time_sec",
        ])

        for dataset_name in args.datasets.split("/"):
            ds_code = ds_alias.get(dataset_name, dataset_name)
            cfg = get_config_file(args.config, ds_code)
            test_loader, classnames, template = build_test_data_loader(ds_code, args.data_root, preprocess, shuffle=False)
            clip_weights = clip_classifier(classnames, template, clip_model, device=device)

            anchor_active = args.enable_anchor_reservoir and isinstance(getattr(clip_model, "visual", None), VisionTransformer)
            cache_tag = f"{backbone_tag}_{dataset_name}_anchor{int(anchor_active)}"
            if args.max_samples is not None:
                cache_tag += f"_n{args.max_samples}"
            cache_path = os.path.join(args.feature_cache_dir, f"{cache_tag}.pt")

            data = None
            if not args.no_cache and os.path.exists(cache_path):
                cached = torch.load(cache_path, map_location=device, weights_only=True)
                data = cached["data"]
                for sample in data:
                    for key, value in sample.items():
                        if isinstance(value, torch.Tensor):
                            sample[key] = value.to(device)
                print(f"Loaded cached features: {cache_path}")

            if data is None:
                data = precompute_features(test_loader, clip_model, clip_weights, device, anchor_active, args.max_samples)
                if not args.no_cache:
                    os.makedirs(args.feature_cache_dir, exist_ok=True)
                    cache_data = []
                    for sample in data:
                        cache_data.append({key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in sample.items()})
                    torch.save({"data": cache_data}, cache_path)
                    print(f"Cached features -> {cache_path}")

            method = f"LC+Anchor{int(anchor_active)}_{args.blend_policy}"
            if args.multi_state:
                method += "_MS"
            start = time.time()
            acc, stats = run_lc_plus_anchor(
                data,
                clip_weights,
                cfg["positive"],
                cfg["negative"],
                device,
                buffer_size=args.buffer_size,
                proto_gamma=args.proto_gamma,
                proto_temperature=args.proto_temperature,
                proto_min_samples=args.proto_min_samples,
                use_multi_state=args.multi_state,
                num_states=args.num_states,
                spawn_threshold=args.spawn_threshold,
                blend_policy=args.blend_policy,
                blend_confidence_gate=args.blend_confidence_gate,
                enable_anchor=anchor_active,
                anchor_capacity=args.anchor_reservoir_size,
                anchor_entropy_threshold=args.anchor_entropy_threshold,
                anchor_alpha=args.anchor_alpha,
                anchor_beta=args.anchor_beta,
                mamba_mode=args.mamba3_mode,
                state_logit_weight=args.state_logit_weight,
            )
            elapsed = time.time() - start
            print(
                f"{dataset_name} {method}: {acc:.2f}% "
                f"[lc={stats['lc_acc']:.2f} p1={stats['pass1_acc']:.2f} "
                f"proto={stats['proto_acc']:.2f} blend={stats['blend_acc']:.2f}] "
                f"used={stats['method_used']} ({elapsed:.1f}s)"
            )
            writer.writerow([
                dataset_name,
                method,
                f"{acc:.2f}",
                f"{stats['lc_acc']:.2f}",
                f"{stats['pass1_acc']:.2f}",
                f"{stats['proto_acc']:.2f}",
                f"{stats['blend_acc']:.2f}",
                stats["method_used"],
                stats["proto_classes"],
                stats["blend_from_proto"],
                f"{stats['anchor_update_accept_rate']:.4f}",
                f"{stats['anchor_fill_ratio']:.4f}",
                f"{stats['avg_anchor_correction_norm']:.4f}",
                stats.get("cache_entries", 0),
                f"{elapsed:.1f}",
            ])
            csv_file.flush()

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
