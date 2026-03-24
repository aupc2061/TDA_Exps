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


class Mamba3Memory:
    def __init__(
        self,
        num_classes,
        feature_dim,
        device,
        include_prob_map=False,
        mode='mamba3-trapezoid',
        decay=0.96,
        min_blend=0.02,
        max_blend=0.35,
        entropy_temp=4.0,
        phase_strength=0.0,
        num_slots=1,
        new_slot_threshold=0.25,
    ):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.include_prob_map = include_prob_map
        self.mode = mode
        self.decay = decay
        self.min_blend = min_blend
        self.max_blend = max_blend
        self.entropy_temp = entropy_temp
        self.phase_strength = phase_strength
        self.num_slots = max(1, int(num_slots))
        self.new_slot_threshold = new_slot_threshold
        self.eps = 1e-8

        valid_modes = {'mamba3-trapezoid', 'mamba3-complex', 'mamba3-multislot'}
        if self.mode not in valid_modes:
            raise ValueError(f'Unknown Mamba3 memory mode: {self.mode}')
        if self.mode != 'mamba3-multislot':
            self.num_slots = 1
        if self.mode == 'mamba3-complex' and self.phase_strength <= 0.0:
            self.phase_strength = 0.15

        self.state_keys = torch.zeros(num_classes, self.num_slots, feature_dim, dtype=torch.float32, device=device)
        self.prev_inputs = torch.zeros(num_classes, self.num_slots, feature_dim, dtype=torch.float32, device=device)
        self.slot_seen = torch.zeros(num_classes, self.num_slots, dtype=torch.bool, device=device)
        self.slot_usage = torch.zeros(num_classes, self.num_slots, dtype=torch.int32, device=device)
        self.last_slot_index = torch.full((num_classes,), -1, dtype=torch.int64, device=device)

        self.state_prob = (
            torch.zeros(num_classes, self.num_slots, num_classes, dtype=torch.float32, device=device)
            if include_prob_map else None
        )
        self.prev_prob = (
            torch.zeros(num_classes, self.num_slots, num_classes, dtype=torch.float32, device=device)
            if include_prob_map else None
        )

        self.total_updates_seen = 0
        self.accepted_updates = 0
        self.last_update_accepted = False
        self.phase_history_sum = 0.0
        self.alpha_history_sum = 0.0
        self.beta_history_sum = 0.0
        self.gamma_history_sum = 0.0

    def is_empty(self):
        return not bool(torch.any(self.slot_seen).item())

    def _compute_delta(self, normalized_entropy, sim):
        confidence = 1.0 - float(normalized_entropy)
        novelty = max(0.0, 1.0 - sim)
        base_gate = torch.sigmoid(
            torch.tensor(confidence * self.entropy_temp, device=self.device, dtype=torch.float32)
        ).item()
        gate = min(1.0, base_gate * novelty)
        return self.min_blend + (self.max_blend - self.min_blend) * gate

    def _compute_confidence_novelty(self, normalized_entropy, sim):
        confidence = min(1.0, max(0.0, 1.0 - float(normalized_entropy)))
        novelty = min(1.0, max(0.0, 1.0 - float(sim)))
        return confidence, novelty

    def _compute_recurrence_coeffs(self, normalized_entropy, sim, update_weight):
        delta_t = self._compute_delta(normalized_entropy, sim) * update_weight
        confidence, novelty = self._compute_confidence_novelty(normalized_entropy, sim)
        decay_rate = 1.0 - self.decay
        alpha_t = torch.exp(torch.tensor(-delta_t * decay_rate, device=self.device)).item()
        current_weight = 0.5 + 0.5 * confidence
        gamma_t = delta_t * current_weight
        beta_t = delta_t * (1.0 - current_weight) * alpha_t
        phase_t = self.phase_strength * novelty * (2.0 * confidence - 1.0)
        return alpha_t, beta_t, gamma_t, phase_t, confidence, novelty

    def _apply_phase_rotation(self, vector, phase):
        if abs(phase) <= self.eps:
            return vector

        rotated = vector.clone()
        pair_dim = (vector.numel() // 2) * 2
        if pair_dim == 0:
            return rotated

        x0 = vector[:pair_dim:2]
        x1 = vector[1:pair_dim:2]
        cos_phase = torch.cos(torch.tensor(phase, device=vector.device, dtype=vector.dtype))
        sin_phase = torch.sin(torch.tensor(phase, device=vector.device, dtype=vector.dtype))
        rotated[:pair_dim:2] = cos_phase * x0 - sin_phase * x1
        rotated[1:pair_dim:2] = sin_phase * x0 + cos_phase * x1
        return rotated

    def _choose_slot(self, pred, obs):
        seen_slots = torch.where(self.slot_seen[pred])[0]
        if seen_slots.numel() == 0:
            return 0

        slot_states = self.state_keys[pred, seen_slots]
        sims = F.cosine_similarity(slot_states, obs.unsqueeze(0), dim=-1)
        best_idx = int(torch.argmax(sims).item())
        best_slot = int(seen_slots[best_idx].item())
        best_sim = float(sims[best_idx].item())

        empty_slots = torch.where(~self.slot_seen[pred])[0]
        if empty_slots.numel() > 0 and best_sim < self.new_slot_threshold:
            return int(empty_slots[0].item())
        return best_slot

    def update(self, pred, feature, normalized_entropy, prob_map=None, update_weight=1.0):
        with torch.no_grad():
            self.total_updates_seen += 1
            self.last_update_accepted = False

            pred = int(pred)
            obs = F.normalize(feature.squeeze(0).float(), dim=-1)
            update_weight = float(max(0.0, min(1.0, update_weight)))
            if update_weight <= 0.0:
                return False

            slot_idx = 0 if self.mode != 'mamba3-multislot' else self._choose_slot(pred, obs)
            prev_state = self.state_keys[pred, slot_idx]
            prev_input = self.prev_inputs[pred, slot_idx]

            if not self.slot_seen[pred, slot_idx]:
                self.state_keys[pred, slot_idx] = obs
                self.prev_inputs[pred, slot_idx] = obs
                self.slot_seen[pred, slot_idx] = True
                self.slot_usage[pred, slot_idx] += 1
                self.last_slot_index[pred] = slot_idx
                if self.include_prob_map and prob_map is not None:
                    prob = prob_map.squeeze(0).float()
                    self.state_prob[pred, slot_idx] = prob
                    self.prev_prob[pred, slot_idx] = prob
                self.accepted_updates += 1
                self.last_update_accepted = True
                return True

            sim = F.cosine_similarity(prev_state.unsqueeze(0), obs.unsqueeze(0)).item()
            alpha_t, beta_t, gamma_t, phase_t, _, _ = self._compute_recurrence_coeffs(
                normalized_entropy,
                sim,
                update_weight,
            )

            if self.mode == 'mamba3-complex':
                prev_state = self._apply_phase_rotation(prev_state, phase_t)
                prev_input = self._apply_phase_rotation(prev_input, phase_t)
                obs = self._apply_phase_rotation(obs, phase_t)

            updated_state = alpha_t * prev_state + beta_t * prev_input + gamma_t * obs
            self.state_keys[pred, slot_idx] = F.normalize(updated_state, dim=-1)
            self.prev_inputs[pred, slot_idx] = obs
            self.slot_usage[pred, slot_idx] += 1
            self.last_slot_index[pred] = slot_idx

            if self.include_prob_map and prob_map is not None:
                obs_prob = prob_map.squeeze(0).float()
                prev_prob = self.prev_prob[pred, slot_idx]
                state_prob = self.state_prob[pred, slot_idx]
                updated_prob = alpha_t * state_prob + beta_t * prev_prob + gamma_t * obs_prob
                updated_prob = updated_prob / (updated_prob.sum() + self.eps)
                self.state_prob[pred, slot_idx] = updated_prob
                self.prev_prob[pred, slot_idx] = obs_prob

            self.accepted_updates += 1
            self.last_update_accepted = True
            self.phase_history_sum += abs(float(phase_t))
            self.alpha_history_sum += float(alpha_t)
            self.beta_history_sum += float(beta_t)
            self.gamma_history_sum += float(gamma_t)
            return True

    def logits(self, image_features, alpha, beta, clip_weights, neg_mask_thresholds=None):
        if self.is_empty():
            return torch.zeros_like(image_features @ clip_weights)

        with torch.no_grad():
            query = F.normalize(image_features.float(), dim=-1)
            flat_keys = self.state_keys.view(self.num_classes * self.num_slots, self.feature_dim)
            flat_seen = self.slot_seen.view(self.num_classes * self.num_slots)
            active_idx = torch.where(flat_seen)[0]
            keys = F.normalize(flat_keys[active_idx], dim=-1)

            if neg_mask_thresholds:
                flat_prob = self.state_prob.view(self.num_classes * self.num_slots, self.num_classes)
                values = flat_prob[active_idx]
                values = (((values > neg_mask_thresholds[0]) & (values < neg_mask_thresholds[1])).type(torch.int8)).to(query.dtype)
            else:
                class_ids = torch.div(active_idx, self.num_slots, rounding_mode='floor')
                values = F.one_hot(class_ids, num_classes=clip_weights.size(1)).to(query.dtype)

            affinity = query @ keys.permute(1, 0)
            memory_logits = ((-1) * (beta - beta * affinity)).exp() @ values.float()
            return alpha * memory_logits.to(image_features.dtype)

    def active_slot_ratio(self):
        return float(self.slot_seen.float().mean().item())

    def avg_slot_utilization(self):
        active = self.slot_usage[self.slot_seen]
        if active.numel() == 0:
            return 0.0
        return float(active.float().mean().item())

    def stats(self):
        accept_rate = float(self.accepted_updates / self.total_updates_seen) if self.total_updates_seen > 0 else 0.0
        denom = max(1, self.accepted_updates - int(torch.sum(self.slot_seen).item()))
        return {
            'accepted_updates': int(self.accepted_updates),
            'total_update_attempts': int(self.total_updates_seen),
            'accept_rate': accept_rate,
            'active_slot_ratio': self.active_slot_ratio(),
            'avg_slot_utilization': self.avg_slot_utilization(),
            'avg_phase_magnitude': self.phase_history_sum / max(1, self.accepted_updates),
            'avg_alpha': self.alpha_history_sum / denom,
            'avg_beta': self.beta_history_sum / denom,
            'avg_gamma': self.gamma_history_sum / denom,
        }

    def memory_bytes(self):
        total_bytes = self.state_keys.element_size() * self.state_keys.nelement()
        total_bytes += self.prev_inputs.element_size() * self.prev_inputs.nelement()
        total_bytes += self.slot_seen.element_size() * self.slot_seen.nelement()
        total_bytes += self.slot_usage.element_size() * self.slot_usage.nelement()
        total_bytes += self.last_slot_index.element_size() * self.last_slot_index.nelement()
        if self.state_prob is not None:
            total_bytes += self.state_prob.element_size() * self.state_prob.nelement()
            total_bytes += self.prev_prob.element_size() * self.prev_prob.nelement()
        return total_bytes


class AnchorReservoir:
    def __init__(self, num_classes, capacity, device, entropy_threshold=0.25):
        self.num_classes = num_classes
        self.capacity = capacity
        self.device = device
        self.entropy_threshold = entropy_threshold
        self.reservoir = {class_idx: [] for class_idx in range(num_classes)}
        self.total_updates_seen = 0
        self.accepted_updates = 0
        self.last_update_accepted = False
        self.invalid_update_rejections = 0
        self.invalid_query_rejections = 0
        self.invalid_anchor_rejections = 0

    def _is_finite_tensor(self, tensor):
        return tensor is not None and bool(torch.isfinite(tensor).all().item())

    def is_empty(self):
        return all(len(items) == 0 for items in self.reservoir.values())

    def update(self, pred, layer_cls_tokens, normalized_entropy, update_weight=1.0):
        self.total_updates_seen += 1
        self.last_update_accepted = False
        if layer_cls_tokens is None or self.capacity <= 0:
            return False
        if not self._is_finite_tensor(layer_cls_tokens):
            self.invalid_update_rejections += 1
            return False

        normalized_entropy = float(normalized_entropy)
        update_weight = float(max(0.0, min(1.0, update_weight)))
        if normalized_entropy > self.entropy_threshold or update_weight <= 0.0:
            return False

        pred = int(pred)
        tokens = torch.nan_to_num(layer_cls_tokens.squeeze(0).detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        tokens = F.normalize(tokens, dim=-1)
        if not self._is_finite_tensor(tokens):
            self.invalid_update_rejections += 1
            return False
        effective_entropy = normalized_entropy / max(update_weight, 1e-6)
        item = [tokens, effective_entropy]

        class_items = self.reservoir[pred]
        class_items.append(item)
        class_items.sort(key=operator.itemgetter(1))
        if len(class_items) > self.capacity:
            del class_items[self.capacity:]
        self.accepted_updates += 1
        self.last_update_accepted = True
        return True

    def logits(self, layer_cls_tokens, alpha, beta):
        if layer_cls_tokens is None or self.is_empty():
            return torch.zeros(1, self.num_classes, device=self.device, dtype=torch.float32)
        if not self._is_finite_tensor(layer_cls_tokens):
            self.invalid_query_rejections += 1
            return torch.zeros(1, self.num_classes, device=self.device, dtype=torch.float32)

        query = torch.nan_to_num(layer_cls_tokens.squeeze(0).float(), nan=0.0, posinf=0.0, neginf=0.0)
        query = F.normalize(query, dim=-1)
        if not self._is_finite_tensor(query):
            self.invalid_query_rejections += 1
            return torch.zeros(1, self.num_classes, device=self.device, dtype=torch.float32)
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
                anchor = torch.nan_to_num(tokens.to(query.device, dtype=query.dtype), nan=0.0, posinf=0.0, neginf=0.0)
                anchor = F.normalize(anchor, dim=-1)
                if not self._is_finite_tensor(anchor):
                    self.invalid_anchor_rejections += 1
                    continue
                per_layer_sim = (query * anchor).sum(dim=-1)
                score = torch.sum(layer_weights * per_layer_sim)
                if not torch.isfinite(score):
                    self.invalid_anchor_rejections += 1
                    continue
                if best_score is None or score > best_score:
                    best_score = score

            if best_score is not None:
                class_scores[class_idx] = best_score

        return alpha * class_scores.unsqueeze(0)

    def fill_ratio(self):
        if self.capacity <= 0 or self.num_classes <= 0:
            return 0.0
        total_items = sum(len(items) for items in self.reservoir.values())
        return float(total_items) / float(self.num_classes * self.capacity)

    def stats(self):
        return {
            'accepted_updates': int(self.accepted_updates),
            'total_update_attempts': int(self.total_updates_seen),
            'accept_rate': float(self.accepted_updates / self.total_updates_seen) if self.total_updates_seen > 0 else 0.0,
            'fill_ratio': self.fill_ratio(),
            'invalid_update_rejections': int(self.invalid_update_rejections),
            'invalid_query_rejections': int(self.invalid_query_rejections),
            'invalid_anchor_rejections': int(self.invalid_anchor_rejections),
        }

    def memory_bytes(self):
        total_bytes = 0
        for class_items in self.reservoir.values():
            for item in class_items:
                tokens = item[0]
                total_bytes += tokens.element_size() * tokens.nelement()
        return total_bytes
