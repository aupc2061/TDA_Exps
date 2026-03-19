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
        correction_mode='heuristic',
        kalman_q=0.01,
        kalman_r=0.05,
        kalman_q_min=0.005,
        kalman_q_max=0.05,
        kalman_r_min=0.01,
        kalman_r_max=0.1,
    ):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.include_prob_map = include_prob_map
        self.decay = decay
        self.min_blend = min_blend
        self.max_blend = max_blend
        self.entropy_temp = entropy_temp
        self.correction_mode = correction_mode
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r
        self.kalman_q_min = kalman_q_min
        self.kalman_q_max = kalman_q_max
        self.kalman_r_min = kalman_r_min
        self.kalman_r_max = kalman_r_max
        self.eps = 1e-8

        valid_modes = {'heuristic', 'kalman-fixed', 'kalman-adaptive', 'kalman-decoupled', 'vmf-fixed', 'vmf-adaptive', 'vmf-online'}
        if self.correction_mode not in valid_modes:
            raise ValueError(f'Unknown SSM correction mode: {self.correction_mode}')

        if self.correction_mode == 'kalman-fixed':
            self.correction_mode = 'vmf-fixed'
        elif self.correction_mode == 'kalman-adaptive':
            self.correction_mode = 'vmf-adaptive'
        elif self.correction_mode == 'kalman-decoupled':
            self.correction_mode = 'vmf-online'

        self.state_keys = torch.zeros(num_classes, feature_dim, dtype=torch.float32, device=device)
        self.seen = torch.zeros(num_classes, dtype=torch.bool, device=device)
        self.state_cov = torch.zeros(num_classes, dtype=torch.float32, device=device)
        self.state_prob = torch.zeros(num_classes, num_classes, dtype=torch.float32, device=device) if include_prob_map else None
        self.state_prob_cov = torch.zeros(num_classes, dtype=torch.float32, device=device) if include_prob_map else None

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

    def _compute_confidence_novelty(self, normalized_entropy, sim):
        confidence = min(1.0, max(0.0, 1.0 - float(normalized_entropy)))
        novelty = min(1.0, max(0.0, 1.0 - float(sim)))
        return confidence, novelty

    def _vmf_concentrations(self, confidence, novelty):
        if self.correction_mode == 'vmf-fixed':
            q = self.kalman_q
            r = self.kalman_r
        else:
            q = self.kalman_q_min + (self.kalman_q_max - self.kalman_q_min) * novelty
            r = self.kalman_r_min + (self.kalman_r_max - self.kalman_r_min) * (1.0 - confidence)

        # Map noise scales to directional precision (higher kappa = more trust).
        kappa_prior = 1.0 / (q + self.eps)
        kappa_obs = 1.0 / (r + self.eps)
        return kappa_prior, kappa_obs

    def _vmf_correct(self, prior_state, observation, confidence, novelty):
        kappa_prior, kappa_obs = self._vmf_concentrations(confidence, novelty)

        prior_dir = F.normalize(prior_state, dim=-1)
        obs_dir = F.normalize(observation, dim=-1)
        eta = kappa_prior * prior_dir + kappa_obs * obs_dir
        post_dir = F.normalize(eta, dim=-1)
        post_kappa = torch.norm(eta, dim=-1)

        return post_dir, post_kappa, kappa_prior, kappa_obs

    def update(self, pred, feature, normalized_entropy, prob_map=None, update_weight=1.0):
        with torch.no_grad():
            pred = int(pred)
            prev = self.state_keys[pred]
            obs = feature.squeeze(0).float()
            update_weight = float(max(0.0, min(1.0, update_weight)))

            if not self.seen[pred]:
                # First time seeing this class, initialize directly
                self.state_keys[pred] = F.normalize(obs, dim=-1)
                self.seen[pred] = True
                self.state_cov[pred] = 1.0
                if self.include_prob_map and prob_map is not None:
                    self.state_prob[pred] = prob_map.squeeze(0).float()
                    self.state_prob_cov[pred] = 1.0
                return

            # --- NOVELTY: Similarity-Aware Selective SSM ---
            # 1. Compute data-dependent step size (Delta) based on Mamba's selective mechanism.
            sim = F.cosine_similarity(prev.unsqueeze(0), obs.unsqueeze(0)).item()
            delta_t = self._compute_delta(normalized_entropy, sim) * update_weight
            confidence, novelty = self._compute_confidence_novelty(normalized_entropy, sim)
            
            # 2. Discretization (Zero-order hold approximation of continuous SSM)
            # Continuous system: h'(t) = -lambda * h(t) + x(t)
            # Discretized: h_t = exp(-Delta * lambda) * h_{t-1} + Delta * x_t
            decay_rate = 1.0 - self.decay
            A_bar = torch.exp(torch.tensor(-delta_t * decay_rate, device=self.state_keys.device)).item()
            B_bar = delta_t

            # Update and normalize to prevent unbounded growth (crucial for TTA stability)
            pred_state = A_bar * prev + B_bar * obs

            if self.correction_mode == 'heuristic':
                self.state_keys[pred] = F.normalize(pred_state, dim=-1)
            else:
                prior_state = A_bar * prev if self.correction_mode == 'vmf-online' else pred_state
                corrected_state, corrected_kappa, kappa_prior, kappa_obs = self._vmf_correct(
                    prior_state,
                    obs,
                    confidence,
                    novelty,
                )
                self.state_keys[pred] = corrected_state
                self.state_cov[pred] = corrected_kappa

            if self.include_prob_map and prob_map is not None:
                prev_prob = self.state_prob[pred]
                obs_prob = prob_map.squeeze(0).float()
                pred_prob = A_bar * prev_prob + B_bar * obs_prob

                if self.correction_mode == 'heuristic':
                    new_prob = pred_prob
                else:
                    prior_prob = A_bar * prev_prob if self.correction_mode == 'vmf-online' else pred_prob
                    w_obs = kappa_obs / (kappa_prior + kappa_obs + self.eps)
                    new_prob = (1.0 - w_obs) * prior_prob + w_obs * obs_prob
                    self.state_prob_cov[pred] = kappa_prior + kappa_obs

                # Normalize probabilities to sum to 1 to keep them within mask thresholds
                self.state_prob[pred] = new_prob / (new_prob.sum() + self.eps)

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
        total_bytes += self.state_cov.element_size() * self.state_cov.nelement()
        if self.state_prob is not None:
            total_bytes += self.state_prob.element_size() * self.state_prob.nelement()
            total_bytes += self.state_prob_cov.element_size() * self.state_prob_cov.nelement()
        return total_bytes


class AnchorReservoir:
    def __init__(self, num_classes, capacity, device, entropy_threshold=0.25):
        self.num_classes = num_classes
        self.capacity = capacity
        self.device = device
        self.entropy_threshold = entropy_threshold
        self.reservoir = {class_idx: [] for class_idx in range(num_classes)}

    def is_empty(self):
        return all(len(items) == 0 for items in self.reservoir.values())

    def update(self, pred, layer_cls_tokens, normalized_entropy, update_weight=1.0):
        if layer_cls_tokens is None or self.capacity <= 0:
            return

        normalized_entropy = float(normalized_entropy)
        update_weight = float(max(0.0, min(1.0, update_weight)))
        if normalized_entropy > self.entropy_threshold or update_weight <= 0.0:
            return

        pred = int(pred)
        tokens = F.normalize(layer_cls_tokens.squeeze(0).detach().float(), dim=-1)
        effective_entropy = normalized_entropy / max(update_weight, 1e-6)
        item = [tokens, effective_entropy]

        class_items = self.reservoir[pred]
        class_items.append(item)
        class_items.sort(key=operator.itemgetter(1))
        if len(class_items) > self.capacity:
            del class_items[self.capacity:]

    def logits(self, layer_cls_tokens, alpha, beta):
        if layer_cls_tokens is None or self.is_empty():
            return torch.zeros(1, self.num_classes, device=self.device, dtype=torch.float32)

        query = F.normalize(layer_cls_tokens.squeeze(0).float(), dim=-1)
        layer_count = query.size(0)
        layer_positions = torch.arange(1, layer_count + 1, device=query.device, dtype=query.dtype)
        layer_weights = torch.exp(beta * layer_positions / float(layer_count))
        layer_weights = layer_weights / layer_weights.sum().clamp_min(1e-6)

        class_scores = torch.zeros(self.num_classes, device=query.device, dtype=query.dtype)
        for class_idx, items in self.reservoir.items():
            if len(items) == 0:
                continue

            best_score = None
            for tokens, _ in items:
                anchor = F.normalize(tokens.to(query.device, dtype=query.dtype), dim=-1)
                per_layer_sim = (query * anchor).sum(dim=-1)
                score = torch.sum(layer_weights * per_layer_sim)
                if best_score is None or score > best_score:
                    best_score = score

            class_scores[class_idx] = best_score

        return alpha * class_scores.unsqueeze(0)

    def memory_bytes(self):
        total_bytes = 0
        for class_items in self.reservoir.values():
            for item in class_items:
                tokens = item[0]
                total_bytes += tokens.element_size() * tokens.nelement()
        return total_bytes
