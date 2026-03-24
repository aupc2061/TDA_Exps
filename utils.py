import os
import yaml
import torch
import math
import numpy as np
import clip
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def resolve_device(device='auto'):
    if isinstance(device, torch.device):
        return device

    requested = str(device).lower()
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available.')
        return torch.device('cuda')

    if requested == 'mps':
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError('MPS requested but not available.')
        return torch.device('mps')

    if requested == 'cpu':
        return torch.device('cpu')

    raise ValueError(f'Unknown device option: {device}')

def get_entropy(loss, clip_weights):
    # Use natural log (math.log) to match x.log_softmax() which uses natural log.
    # This ensures normalized_entropy is strictly bounded between [0, 1].
    max_entropy = math.log(clip_weights.size(1))
    return float(loss / max_entropy)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = correct[: topk].reshape(-1).float().sum().item()
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model, device=None):
    if device is None:
        device = next(clip_model.parameters()).device

    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(device)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights


def get_clip_logits(images, clip_model, clip_weights, device=None):
    if device is None:
        device = next(clip_model.parameters()).device

    with torch.no_grad():
        if isinstance(images, list):
            images = torch.cat(images, dim=0).to(device)
        else:
            images = images.to(device)

        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        clip_logits = 100. * image_features @ clip_weights

        if image_features.size(0) > 1:
            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
            output = clip_logits[selected_idx]
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            clip_logits = output.mean(0).unsqueeze(0)

            loss = avg_entropy(output)
            prob_map = output.softmax(1).mean(0).unsqueeze(0)
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        else:
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

        return image_features, clip_logits, loss, prob_map, pred


def get_clip_logits_with_details(images, clip_model, clip_weights, device=None, return_layer_cls=False):
    if device is None:
        device = next(clip_model.parameters()).device

    with torch.no_grad():
        if isinstance(images, list):
            images = torch.cat(images, dim=0).to(device)
        else:
            images = images.to(device)

        model_outputs = clip_model.encode_image(images, return_layer_cls=return_layer_cls)
        if isinstance(model_outputs, dict):
            raw_image_features = model_outputs["image_features"]
            layer_cls_tokens = model_outputs.get("layer_cls_tokens")
        else:
            raw_image_features = model_outputs
            layer_cls_tokens = None

        image_features = raw_image_features / raw_image_features.norm(dim=-1, keepdim=True)
        clip_logits = 100. * image_features @ clip_weights

        if image_features.size(0) > 1:
            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
            output = clip_logits[selected_idx]
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            clip_logits = output.mean(0).unsqueeze(0)
            selected_layer_cls = None if layer_cls_tokens is None else layer_cls_tokens[selected_idx]
            if selected_layer_cls is not None:
                selected_layer_cls = selected_layer_cls.mean(dim=0, keepdim=True)

            loss = avg_entropy(output)
            prob_map = output.softmax(1).mean(0).unsqueeze(0)
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        else:
            selected_layer_cls = layer_cls_tokens
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

        details = {
            'layer_cls_tokens': selected_layer_cls,
            'supports_anchor': selected_layer_cls is not None,
        }

        return image_features, clip_logits, loss, prob_map, pred, details


def get_ood_preprocess():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

    return aug_preprocess


def get_consensus_preprocess(n_views=8):
    """Create AugMixAugmenter for multi-view consensus gating."""
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    return AugMixAugmenter(base_transform, preprocess, n_views=n_views, augmix=True)


def _apply_top_k_mask(logits, top_k):
    num_classes = logits.size(1)
    k = min(top_k, num_classes)
    _, topk_idx = logits.topk(k, dim=1)
    masked_logits = torch.full_like(logits, float('-inf'))
    masked_logits.scatter_(1, topk_idx, logits.gather(1, topk_idx))
    return masked_logits


def _normalized_entropy_from_probs(probs):
    eps = 1e-8
    entropy = -(probs * torch.log(probs + eps)).sum(dim=1)
    return entropy / math.log(probs.size(1))


def _distribution_margin(probs):
    if probs.size(1) < 2:
        return 1.0
    top2 = probs.topk(min(2, probs.size(1)), dim=1).values
    return float((top2[:, 0] - top2[:, 1]).mean().item())


def _encode_views(view_tensors, clip_model, clip_weights, device, return_layer_cls=False):
    image_batch = torch.cat(view_tensors, dim=0).to(device)
    model_outputs = clip_model.encode_image(image_batch, return_layer_cls=return_layer_cls)
    if isinstance(model_outputs, dict):
        raw_features = model_outputs["image_features"]
        layer_cls_tokens = model_outputs.get("layer_cls_tokens")
    else:
        raw_features = model_outputs
        layer_cls_tokens = None

    features = raw_features
    features = features / features.norm(dim=-1, keepdim=True)
    logits = 100.0 * features @ clip_weights
    return features, logits, layer_cls_tokens


def _compute_soft_consensus_outputs(all_features, all_logits, clip_weights, top_k,
                                    consensus_threshold, device, all_layer_cls_tokens=None):
    masked_logits = _apply_top_k_mask(all_logits, top_k)
    masked_probs = masked_logits.softmax(dim=1)
    mean_prob = masked_probs.mean(dim=0, keepdim=True)

    eps = 1e-8
    log_mean_prob = torch.log(mean_prob + eps)
    log_masked_probs = torch.log(masked_probs + eps)
    per_view_kl = (masked_probs * (log_masked_probs - log_mean_prob)).sum(dim=1)
    js_div = per_view_kl.mean()
    agreement_score = 1.0 - min(1.0, float(js_div.item() / math.log(clip_weights.size(1))))
    confidence_score = 1.0 - float(_normalized_entropy_from_probs(mean_prob).item())
    soft_score = max(0.0, min(1.0, agreement_score * confidence_score))

    score_temperature = 0.10
    update_weight = torch.sigmoid(
        torch.tensor((soft_score - consensus_threshold) / score_temperature, device=device, dtype=torch.float32)
    ).item()
    soft_accept = soft_score >= consensus_threshold

    view_weight_temperature = 0.10
    view_weights = torch.softmax(-per_view_kl / view_weight_temperature, dim=0)
    denoised_feature = (view_weights.unsqueeze(1) * all_features).sum(dim=0, keepdim=True)
    denoised_feature = denoised_feature / denoised_feature.norm(dim=-1, keepdim=True)
    denoised_layer_cls = None
    if all_layer_cls_tokens is not None:
        denoised_layer_cls = (view_weights.view(-1, 1, 1) * all_layer_cls_tokens).sum(dim=0, keepdim=True)
        denoised_layer_cls = denoised_layer_cls / denoised_layer_cls.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    pred = int(mean_prob.argmax(dim=1).item())

    denoised_logits = 100.0 * denoised_feature @ clip_weights
    loss = -(mean_prob * torch.log(mean_prob + eps)).sum(dim=1)
    prob_map = mean_prob

    stats = {
        'soft_score': soft_score,
        'agreement_score': agreement_score,
        'confidence_score': confidence_score,
        'update_weight': update_weight,
        'soft_accept': soft_accept,
        'margin': _distribution_margin(mean_prob),
        'num_views_used': int(all_features.size(0)),
    }

    details = {
        'layer_cls_tokens': denoised_layer_cls,
        'supports_anchor': denoised_layer_cls is not None,
    }

    return denoised_feature, denoised_logits, loss, prob_map, pred, stats, details


def dynamic_multiview_consensus_logits(images, clip_model, clip_weights, top_k=5,
                                       consensus_threshold=0.5, dynamic_kwargs=None,
                                       device=None, return_layer_cls=False):
    if device is None:
        device = next(clip_model.parameters()).device

    if not isinstance(images, list) or len(images) <= 1:
        return multiview_consensus_logits(
            images,
            clip_model,
            clip_weights,
            top_k=top_k,
            consensus_threshold=consensus_threshold,
            device=device,
            return_layer_cls=return_layer_cls,
        )

    dynamic_kwargs = dynamic_kwargs or {}
    available_total = len(images)
    max_budget = min(int(dynamic_kwargs.get('max_views', available_total)), available_total)
    max_budget = max(1, max_budget)
    min_budget = min(max(2, int(dynamic_kwargs.get('min_views', 2))), max_budget)
    mid_budget = min(max(min_budget, int(dynamic_kwargs.get('mid_views', 4))), max_budget)

    single_view_confidence = float(dynamic_kwargs.get('single_view_confidence', 0.80))
    single_view_margin = float(dynamic_kwargs.get('single_view_margin', 0.35))
    early_stop_score = float(dynamic_kwargs.get('early_stop_score', 0.60))
    early_stop_margin = float(dynamic_kwargs.get('early_stop_margin', 0.25))
    medium_score = float(dynamic_kwargs.get('medium_score', 0.45))
    medium_margin = float(dynamic_kwargs.get('medium_margin', 0.18))
    plateau_delta = float(dynamic_kwargs.get('plateau_delta', 0.03))
    plateau_margin = float(dynamic_kwargs.get('plateau_margin', 0.15))

    encoded_features = []
    encoded_logits = []
    encoded_layer_cls = []
    views_used = 0

    def _extend_to(target_budget):
        nonlocal views_used
        target_budget = min(target_budget, max_budget)
        if target_budget <= views_used:
            all_features = torch.cat(encoded_features, dim=0)
            all_logits = torch.cat(encoded_logits, dim=0)
            all_layer_cls = None if len(encoded_layer_cls) == 0 else torch.cat(encoded_layer_cls, dim=0)
            return _compute_soft_consensus_outputs(
                all_features, all_logits, clip_weights, top_k, consensus_threshold, device, all_layer_cls_tokens=all_layer_cls
            )

        new_features, new_logits, new_layer_cls = _encode_views(
            images[views_used:target_budget],
            clip_model,
            clip_weights,
            device,
            return_layer_cls=return_layer_cls,
        )
        encoded_features.append(new_features)
        encoded_logits.append(new_logits)
        if new_layer_cls is not None:
            encoded_layer_cls.append(new_layer_cls)
        views_used = target_budget
        all_features = torch.cat(encoded_features, dim=0)
        all_logits = torch.cat(encoded_logits, dim=0)
        all_layer_cls = None if len(encoded_layer_cls) == 0 else torch.cat(encoded_layer_cls, dim=0)
        return _compute_soft_consensus_outputs(
            all_features, all_logits, clip_weights, top_k, consensus_threshold, device, all_layer_cls_tokens=all_layer_cls
        )

    result = _extend_to(1)
    stats = result[-1]
    if stats['confidence_score'] >= single_view_confidence and stats['margin'] >= single_view_margin:
        stats['stop_reason'] = 'single_view_high_confidence'
        stats['max_views_available'] = max_budget
        stats['view_savings'] = max_budget - stats['num_views_used']
        return result

    result = _extend_to(min_budget)
    stats = result[-1]
    if stats['soft_score'] >= early_stop_score and stats['margin'] >= early_stop_margin:
        stats['stop_reason'] = 'early_stop_strong_agreement'
        stats['max_views_available'] = max_budget
        stats['view_savings'] = max_budget - stats['num_views_used']
        return result

    prev_soft_score = stats['soft_score']
    target_budget = mid_budget if (stats['soft_score'] >= medium_score or stats['margin'] >= medium_margin) else max_budget

    if target_budget > views_used:
        result = _extend_to(target_budget)
        stats = result[-1]

    if views_used < max_budget:
        score_delta = abs(stats['soft_score'] - prev_soft_score)
        stable_enough = stats['soft_score'] >= medium_score and stats['margin'] >= plateau_margin and score_delta <= plateau_delta
        if stable_enough:
            stats['stop_reason'] = 'mid_budget_plateau'
            stats['max_views_available'] = max_budget
            stats['view_savings'] = max_budget - stats['num_views_used']
            return result

        result = _extend_to(max_budget)
        stats = result[-1]

    stats['stop_reason'] = 'max_budget'
    stats['max_views_available'] = max_budget
    stats['view_savings'] = max_budget - stats['num_views_used']
    return result


def multiview_consensus_logits(images, clip_model, clip_weights, top_k=5,
                               consensus_threshold=0.5, device=None, return_layer_cls=False):
    """
    Process multiple augmented views with top-K masking and consensus gating.

    Steps:
      1. Encode all views independently through CLIP.
      2. Apply top-K masking to each view's logits (remove noisy tail).
      3. Check if >= consensus_threshold of views agree on argmax.
      4. If consensus: denoised feature = mean of agreeing views' features.
      5. If no consensus: feature = mean of all views' features.

    Returns:
        image_features  [1, D]   - denoised feature vector
        clip_logits     [1, C]   - logits from denoised features
        loss            scalar   - softmax entropy of denoised logits
        prob_map        [1, C]   - softmax of denoised logits
        pred            int      - predicted class index
        stats           dict     - soft agreement diagnostics and update weight
    """
    if device is None:
        device = next(clip_model.parameters()).device

    with torch.no_grad():
        if isinstance(images, list):
            all_features, all_logits, all_layer_cls = _encode_views(
                images, clip_model, clip_weights, device, return_layer_cls=return_layer_cls
            )
        else:
            all_features, all_logits, all_layer_cls = _encode_views(
                [images], clip_model, clip_weights, device, return_layer_cls=return_layer_cls
            )

        return _compute_soft_consensus_outputs(
            all_features,
            all_logits,
            clip_weights,
            top_k,
            consensus_threshold,
            device,
            all_layer_cls_tokens=all_layer_cls,
        )


def get_config_file(config_path, dataset_name):
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "S"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    else:
        config_name = f"{dataset_name}.yaml"
    
    config_file = os.path.join(config_path, config_name)
    
    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    return cfg


def build_test_data_loader(dataset_name, root_path, preprocess, shuffle=True, n_views=0):
    if n_views > 0:
        tfm = get_consensus_preprocess(n_views)
    else:
        tfm = preprocess

    if dataset_name == 'I':
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=1, num_workers=8, shuffle=shuffle)
    
    elif dataset_name in ['A','V','R','S']:
        if n_views == 0:
            tfm = get_ood_preprocess()
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=tfm, shuffle=shuffle)

    elif dataset_name in ['caltech101','dtd','eurosat','fgvc','food101','oxford_flowers','oxford_pets','stanford_cars','sun397','ucf101']:
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=tfm, shuffle=shuffle)
    
    else:
        raise "Dataset is not from the chosen list"
    
    return test_loader, dataset.classnames, dataset.template
