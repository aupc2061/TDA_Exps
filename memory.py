import operator

import torch
import torch.nn.functional as F


class CacheMemory:
    def __init__(self, shot_capacity, include_prob_map=False):
        self.shot_capacity = shot_capacity
        self.include_prob_map = include_prob_map
        self.cache = {}

    def is_empty(self):
        return len(self.cache) == 0

    def update(self, pred, feature, loss, prob_map=None):
        with torch.no_grad():
            item = [feature, loss] if not self.include_prob_map else [feature, loss, prob_map]
            if pred in self.cache:
                if len(self.cache[pred]) < self.shot_capacity:
                    self.cache[pred].append(item)
                elif loss < self.cache[pred][-1][1]:
                    self.cache[pred][-1] = item
                self.cache[pred] = sorted(self.cache[pred], key=operator.itemgetter(1))
            else:
                self.cache[pred] = [item]

    def logits(self, image_features, alpha, beta, clip_weights, neg_mask_thresholds=None):
        if self.is_empty():
            return torch.zeros_like(image_features @ clip_weights)

        with torch.no_grad():
            cache_keys, cache_values = [], []
            for class_index in sorted(self.cache.keys()):
                for item in self.cache[class_index]:
                    cache_keys.append(item[0])
                    if neg_mask_thresholds:
                        cache_values.append(item[2])
                    else:
                        cache_values.append(class_index)

            cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
            if neg_mask_thresholds:
                cache_values = torch.cat(cache_values, dim=0)
                cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).to(image_features.device).to(image_features.dtype)
            else:
                cache_values = F.one_hot(torch.tensor(cache_values, dtype=torch.int64, device=image_features.device), num_classes=clip_weights.size(1)).to(image_features.dtype)

            affinity = image_features @ cache_keys
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            return alpha * cache_logits

    def memory_bytes(self):
        total_bytes = 0
        for class_items in self.cache.values():
            for item in class_items:
                for tensor in item:
                    if isinstance(tensor, torch.Tensor):
                        total_bytes += tensor.element_size() * tensor.nelement()
        return total_bytes


class SSMemory:
    def __init__(
        self,
        num_classes,
        feature_dim,
        device,
        include_prob_map=False,
        decay=0.96,
        min_blend=0.02,
        max_blend=0.35,
        entropy_temp=4.0,
    ):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.include_prob_map = include_prob_map
        self.decay = decay
        self.min_blend = min_blend
        self.max_blend = max_blend
        self.entropy_temp = entropy_temp

        self.state_keys = torch.zeros(num_classes, feature_dim, dtype=torch.float32, device=device)
        self.seen = torch.zeros(num_classes, dtype=torch.bool, device=device)
        self.state_prob = torch.zeros(num_classes, num_classes, dtype=torch.float32, device=device) if include_prob_map else None

    def is_empty(self):
        return not bool(torch.any(self.seen).item())

    def _compute_delta(self, normalized_entropy, sim):
        confidence = 1.0 - float(normalized_entropy)
        # Novelty measure: 1.0 - cosine similarity. 
        # If the feature is identical to the state, novelty is 0.
        novelty = max(0.0, 1.0 - sim)
        
        # Base confidence gate (similar to standard selective SSM)
        base_gate = torch.sigmoid(torch.tensor(confidence * self.entropy_temp, device=self.state_keys.device, dtype=torch.float32)).item()
        
        # Modulate the gate by novelty. This is our training-free proxy for Mamba's input-dependent Delta.
        gate = min(1.0, base_gate * novelty)
        
        return self.min_blend + (self.max_blend - self.min_blend) * gate

    def update(self, pred, feature, normalized_entropy, prob_map=None):
        with torch.no_grad():
            pred = int(pred)
            prev = self.state_keys[pred]
            obs = feature.squeeze(0).float()

            if not self.seen[pred]:
                # First time seeing this class, initialize directly
                self.state_keys[pred] = obs
                self.seen[pred] = True
                if self.include_prob_map and prob_map is not None:
                    self.state_prob[pred] = prob_map.squeeze(0).float()
                return

            # --- NOVELTY: Similarity-Aware Selective SSM ---
            # 1. Compute data-dependent step size (Delta) based on Mamba's selective mechanism.
            sim = F.cosine_similarity(prev.unsqueeze(0), obs.unsqueeze(0)).item()
            delta_t = self._compute_delta(normalized_entropy, sim)
            
            # 2. Discretization (Zero-order hold approximation of continuous SSM)
            # Continuous system: h'(t) = -lambda * h(t) + x(t)
            # Discretized: h_t = exp(-Delta * lambda) * h_{t-1} + Delta * x_t
            decay_rate = 1.0 - self.decay
            A_bar = torch.exp(torch.tensor(-delta_t * decay_rate, device=self.state_keys.device)).item()
            B_bar = delta_t

            # Update and normalize to prevent unbounded growth (crucial for TTA stability)
            new_state = A_bar * prev + B_bar * obs
            self.state_keys[pred] = F.normalize(new_state, dim=-1)

            if self.include_prob_map and prob_map is not None:
                prev_prob = self.state_prob[pred]
                obs_prob = prob_map.squeeze(0).float()
                new_prob = A_bar * prev_prob + B_bar * obs_prob
                # Normalize probabilities to sum to 1 to keep them within mask thresholds
                self.state_prob[pred] = new_prob / new_prob.sum()

    def logits(self, image_features, alpha, beta, clip_weights, neg_mask_thresholds=None):
        if self.is_empty():
            return torch.zeros_like(image_features @ clip_weights)

        with torch.no_grad():
            active_idx = torch.where(self.seen)[0]
            keys = F.normalize(self.state_keys[active_idx], dim=-1).permute(1, 0)

            if neg_mask_thresholds:
                prob_values = self.state_prob[active_idx]
                values = (((prob_values > neg_mask_thresholds[0]) & (prob_values < neg_mask_thresholds[1])).type(torch.int8)).to(image_features.dtype)
            else:
                values = F.one_hot(active_idx, num_classes=clip_weights.size(1)).to(image_features.dtype)

            affinity = image_features.float() @ keys
            memory_logits = ((-1) * (beta - beta * affinity)).exp() @ values.float()
            return alpha * memory_logits.to(image_features.dtype)

    def memory_bytes(self):
        total_bytes = self.state_keys.element_size() * self.state_keys.nelement()
        total_bytes += self.seen.element_size() * self.seen.nelement()
        if self.state_prob is not None:
            total_bytes += self.state_prob.element_size() * self.state_prob.nelement()
        return total_bytes