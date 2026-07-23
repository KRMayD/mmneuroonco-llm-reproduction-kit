from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from loss_presets import get_atomic_loss_preset

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss
    
class HardNegativeLoss(nn.Module):
    """
    Hard Negative Noise Contrastive Estimation proposed in https://arxiv.org/abs/2301.02280
    beta1: hardness parameter for image features
    beta2: hardness parameter for text features
    alpha: the weighting function of the positive sample loss
    Setting alpha to 0, the loss is equivalent to the decoupled HN-NCE loss (DHN-NCE)
    temperature: temperature to control the sharpness of the distribution
    """
    def __init__(self, temperature=1.0,beta1=1.0, beta2 = 1.0, alpha=0.0, batch_size=1):
        super(HardNegativeLoss, self).__init__()
        self.temperature = temperature
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.batch_size = batch_size

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Compute cosine similarity between image and text features
        logits_per_image = torch.matmul(image_features, text_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()

        mask = torch.eye(logits_per_image.size(0), dtype=torch.bool)
        mask = mask.to(image_features.device)

        # Positive pairs: diagonal elements
        pos = torch.exp(logits_per_image*mask)

        # Negative pairs: off-diagonal elements
        N = self.batch_size - 1

        neg_mask = ~mask

        # Calculate reweighting factors
        norm_term_img = torch.sum(torch.exp(logits_per_image*neg_mask),dim=-1)
        reweight_img = N * (torch.exp(self.beta1*logits_per_image*neg_mask))/norm_term_img
        norm_term_text = torch.sum(torch.exp(logits_per_text*neg_mask),dim=-1)
        reweight_text = N * (torch.exp(self.beta2*logits_per_text*neg_mask))/norm_term_text

        neg_img = reweight_img * torch.exp(logits_per_image*neg_mask)
        neg_text = reweight_text * torch.exp(logits_per_text*neg_mask)

        # Calculate loss
        loss = -torch.log(pos / (pos*self.alpha + neg_img)) -torch.log(pos / (pos*self.alpha + neg_text))

        return {"contrastive_loss": loss.mean()} if output_dict else loss.mean()


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class dpo_DpoLoss(nn.Module):
    def __init__(
            self,
            alpha: float = 1.0,
            beta: float = 1.0,
            loss_type: str = "4way",
            fourway_weights=None,
            normalize_features: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.loss_type = loss_type
        self.normalize_features = normalize_features
        if fourway_weights is None:
            fourway_weights = (1.0, 1.0, 1.0, 1.0)
        if len(fourway_weights) != 4:
            raise ValueError("fourway_weights must contain exactly 4 values")
        if any(float(weight) < 0 for weight in fourway_weights):
            raise ValueError("fourway_weights must be non-negative")
        if sum(float(weight) for weight in fourway_weights) <= 0:
            raise ValueError("fourway_weights must sum to a positive value")
        self.register_buffer(
            "fourway_weights",
            torch.tensor([float(weight) for weight in fourway_weights], dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        image_features_pos,
        text_features_pos,
        image_features_neg,
        text_features_neg,
        ref_image_features_pos,
        ref_text_features_pos,
        ref_image_features_neg,
        ref_text_features_neg,
        output_dict: bool = False,
    ):
        def score(image_features, text_features):
            if self.normalize_features:
                return self.alpha * F.cosine_similarity(image_features, text_features, dim=-1)
            return self.alpha * (image_features * text_features).sum(dim=-1)

        s_theta_pos1 = score(image_features_pos, text_features_pos)
        s_theta_pos2 = score(image_features_neg, text_features_neg)
        s_theta_neg1 = score(image_features_pos, text_features_neg)
        s_theta_neg2 = score(image_features_neg, text_features_pos)

        s_ref_pos1 = score(ref_image_features_pos, ref_text_features_pos)
        s_ref_pos2 = score(ref_image_features_neg, ref_text_features_neg)
        s_ref_neg1 = score(ref_image_features_pos, ref_text_features_neg)
        s_ref_neg2 = score(ref_image_features_neg, ref_text_features_pos)

        def dpo_func(s_th_p, s_th_n, s_r_p, s_r_n):
            logits = self.beta * ((s_th_p - s_th_n) - (s_r_p - s_r_n))
            return -torch.log(torch.sigmoid(logits))

        if self.loss_type == "text":
            loss_vec = dpo_func(s_theta_pos1, s_theta_neg1, s_ref_pos1, s_ref_neg1)
            loss = loss_vec.mean()
        elif self.loss_type == "image":
            loss_vec = dpo_func(s_theta_pos1, s_theta_neg2, s_ref_pos1, s_ref_neg2)
            loss = loss_vec.mean()
        else:
            loss_terms = torch.stack(
                (
                    dpo_func(s_theta_pos1, s_theta_neg1, s_ref_pos1, s_ref_neg1),
                    dpo_func(s_theta_pos2, s_theta_neg2, s_ref_pos2, s_ref_neg2),
                    dpo_func(s_theta_pos1, s_theta_neg2, s_ref_pos1, s_ref_neg2),
                    dpo_func(s_theta_pos2, s_theta_neg1, s_ref_pos2, s_ref_neg1),
                ),
                dim=0,
            )
            weights = self.fourway_weights.to(device=loss_terms.device, dtype=loss_terms.dtype)
            loss_vec = (loss_terms * weights[:, None]).sum(dim=0)
            loss = loss_vec.mean()

        if output_dict:
            return {"dpo_loss": loss}
        return loss


@dataclass
class dpo_AtomicDpoInputs:
    image_pos: torch.Tensor
    image_neg: torch.Tensor
    text_pos: torch.Tensor
    text_neg: torch.Tensor
    ref_image_pos: torch.Tensor
    ref_image_neg: torch.Tensor
    ref_text_pos: torch.Tensor
    ref_text_neg: torch.Tensor
    atomic_text_pos: Optional[torch.Tensor] = None
    atomic_text_neg: Optional[torch.Tensor] = None
    ref_atomic_text_pos: Optional[torch.Tensor] = None
    ref_atomic_text_neg: Optional[torch.Tensor] = None
    atomic_image_neg: Optional[torch.Tensor] = None
    ref_atomic_image_neg: Optional[torch.Tensor] = None
    atomic_mask: Optional[torch.Tensor] = None
    atomic_text_weight: Optional[torch.Tensor] = None
    atomic_attr_idx: Optional[torch.Tensor] = None
    atomic_visual_neg_found_mask: Optional[torch.Tensor] = None
    atomic_visual_neg_mismatch_count: Optional[torch.Tensor] = None
    effective_atomic_mask: Optional[torch.Tensor] = None

    @property
    def active_atomic_mask(self):
        if self.effective_atomic_mask is not None:
            return self.effective_atomic_mask
        return self.atomic_mask


def combine_optional_masks(*masks):
    active_mask = None
    for mask in masks:
        if mask is None:
            continue
        bool_mask = mask.to(dtype=torch.bool)
        active_mask = bool_mask if active_mask is None else (active_mask & bool_mask)
    return active_mask


def weighted_reduce(loss_tensor, mask=None, weight=None):
    combined_weight = torch.ones_like(loss_tensor)

    if mask is not None:
        combined_weight = combined_weight * mask.to(
            device=loss_tensor.device,
            dtype=loss_tensor.dtype,
        )

    if weight is not None:
        combined_weight = combined_weight * weight.to(
            device=loss_tensor.device,
            dtype=loss_tensor.dtype,
        )

    denom = combined_weight.sum().clamp_min(1.0)
    return (loss_tensor * combined_weight).sum() / denom


def summarize_weight_stats(weight_tensor, mask=None):
    if weight_tensor is None:
        return None

    weight_tensor = weight_tensor.to(dtype=torch.float32)
    if mask is not None:
        active = mask.to(device=weight_tensor.device, dtype=torch.bool)
    else:
        active = torch.ones_like(weight_tensor, dtype=torch.bool)

    active_values = weight_tensor[active]
    if active_values.numel() == 0:
        zero = weight_tensor.new_zeros(())
        return zero, zero, zero
    return active_values.mean(), active_values.max(), active_values.min()


def apply_atomic_dropout_mask(atomic_mask, training, atomic_dropout_k=0, atomic_dropout_p=0.0):
    if atomic_mask is None or atomic_mask.dim() != 2 or not training:
        return atomic_mask

    if atomic_dropout_k <= 0 and atomic_dropout_p <= 0:
        return atomic_mask

    source_mask = atomic_mask.to(dtype=torch.bool)
    mask = source_mask.clone()

    if atomic_dropout_k > 0:
        batch_size, _ = mask.shape
        new_mask = torch.zeros_like(mask)
        for batch_idx in range(batch_size):
            valid_idx = torch.nonzero(source_mask[batch_idx], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue
            keep_count = min(atomic_dropout_k, valid_idx.numel())
            chosen = valid_idx[torch.randperm(valid_idx.numel(), device=mask.device)[:keep_count]]
            new_mask[batch_idx, chosen] = True
        mask = new_mask
    elif atomic_dropout_p > 0:
        keep_prob = 1.0 - atomic_dropout_p
        keep = torch.bernoulli(
            torch.full(source_mask.shape, keep_prob, device=source_mask.device, dtype=torch.float32)
        ).to(dtype=torch.bool)
        mask = source_mask & keep

        empty_rows = mask.sum(dim=1) == 0
        if empty_rows.any():
            for batch_idx in torch.nonzero(empty_rows, as_tuple=False).flatten():
                valid_idx = torch.nonzero(source_mask[batch_idx], as_tuple=False).flatten()
                if valid_idx.numel() == 0:
                    continue
                chosen = valid_idx[torch.randint(valid_idx.numel(), (1,), device=mask.device)]
                mask[batch_idx, chosen] = True

    return mask.to(dtype=atomic_mask.dtype)


class dpo_DpoCore(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, normalize_features: bool = True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.normalize_features = normalize_features

    def score(self, image_features, text_features):
        if self.normalize_features:
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

        if image_features.dim() == 2 and text_features.dim() == 2:
            if self.normalize_features:
                return self.alpha * F.cosine_similarity(image_features, text_features, dim=-1)
            return self.alpha * (image_features * text_features).sum(dim=-1)
        if image_features.dim() == 2 and text_features.dim() == 3:
            return self.alpha * torch.einsum("bd,bad->ba", image_features, text_features)
        if image_features.dim() == 3 and text_features.dim() == 2:
            return self.alpha * torch.einsum("bad,bd->ba", image_features, text_features)
        if image_features.dim() == 3 and text_features.dim() == 3:
            return self.alpha * (image_features * text_features).sum(dim=-1)
        raise ValueError(
            f"Unsupported score input shapes: image={image_features.shape}, text={text_features.shape}"
        )

    def dpo_pair_loss(self, s_theta_pos, s_theta_neg, s_ref_pos, s_ref_neg):
        logits = self.beta * ((s_theta_pos - s_theta_neg) - (s_ref_pos - s_ref_neg))
        return -F.logsigmoid(logits)


class dpo_UniformWeight(nn.Module):
    def forward(self, **kwargs):
        return None


class dpo_AttrPriorWeight(nn.Module):
    def __init__(self, attr_weights=None):
        super().__init__()
        if attr_weights is not None:
            attr_weights = torch.as_tensor(attr_weights, dtype=torch.float32)
            self.register_buffer("attr_weights", attr_weights)
        else:
            self.attr_weights = None

    def forward(self, x: dpo_AtomicDpoInputs, loss_mat, **kwargs):
        if self.attr_weights is None:
            raise ValueError("attr_weights are required for attr_prior text weighting")

        attr_weights = self.attr_weights.to(device=loss_mat.device, dtype=loss_mat.dtype)
        if loss_mat.dim() == 2:
            return attr_weights.view(1, -1).expand_as(loss_mat)
        if loss_mat.dim() == 1:
            if x.atomic_attr_idx is None:
                raise ValueError("attr_idx is required for 1D atomic tensors when attr_prior is enabled")
            return attr_weights[x.atomic_attr_idx.to(device=loss_mat.device)]
        raise ValueError(f"Unsupported target tensor shape for attr prior weights: {loss_mat.shape}")


class dpo_SemanticRefCacheWeight(nn.Module):
    def forward(self, x: dpo_AtomicDpoInputs, loss_mat, **kwargs):
        if x.atomic_text_weight is None:
            raise ValueError("semantic_ref_cache mode requires atomic_text_weight from the dataset/cache")
        return x.atomic_text_weight.to(device=loss_mat.device, dtype=loss_mat.dtype)


class dpo_SemanticRefOnlineWeight(nn.Module):
    def __init__(self, scale=1.0, weight_min=0.5, weight_max=2.0):
        super().__init__()
        self.scale = scale
        self.weight_min = weight_min
        self.weight_max = weight_max

    def forward(self, x: dpo_AtomicDpoInputs, loss_mat, **kwargs):
        if x.ref_atomic_text_pos is None or x.ref_atomic_text_neg is None:
            raise ValueError("semantic_ref_online mode requires reference atomic text features")

        pos = F.normalize(x.ref_atomic_text_pos, dim=-1)
        neg = F.normalize(x.ref_atomic_text_neg, dim=-1)
        similarity = (pos * neg).sum(dim=-1)
        hardness = (similarity + 1.0) / 2.0
        weight = self.weight_min + (self.weight_max - self.weight_min) * hardness
        return weight.clamp(self.weight_min, self.weight_max).to(
            device=loss_mat.device,
            dtype=loss_mat.dtype,
        )


class dpo_DynamicMarginWeight(nn.Module):
    def __init__(self, hard_mining="none", temperature=0.1, hard_topk=0, weight_min=0.5, weight_max=2.0):
        super().__init__()
        self.hard_mining = hard_mining
        self.temperature = temperature
        self.hard_topk = hard_topk
        self.weight_min = weight_min
        self.weight_max = weight_max

    def forward(self, x: dpo_AtomicDpoInputs, s_th_pos, s_th_neg, s_ref_pos, s_ref_neg, **kwargs):
        if self.hard_mining == "none":
            return None

        with torch.no_grad():
            dpo_margin = (s_th_pos - s_th_neg) - (s_ref_pos - s_ref_neg)

            if self.hard_mining == "margin_weight":
                weight = torch.sigmoid(-dpo_margin / max(self.temperature, 1e-6))
                if x.active_atomic_mask is not None:
                    active = x.active_atomic_mask.to(device=weight.device, dtype=weight.dtype)
                    mean = (weight * active).sum() / active.sum().clamp_min(1.0)
                    weight = weight / mean.clamp_min(1e-6)
                    weight = weight.clamp(self.weight_min, self.weight_max)
                    weight = weight * active
                else:
                    weight = weight / weight.mean().clamp_min(1e-6)
                    weight = weight.clamp(self.weight_min, self.weight_max)
                return weight

            if self.hard_mining == "topk":
                flat_margin = dpo_margin.flatten()
                if x.active_atomic_mask is not None:
                    flat_valid = x.active_atomic_mask.flatten() > 0
                    valid_idx = torch.nonzero(flat_valid, as_tuple=False).flatten()
                else:
                    valid_idx = torch.arange(flat_margin.numel(), device=flat_margin.device)

                keep_count = min(self.hard_topk, valid_idx.numel())
                if keep_count <= 0:
                    return None

                valid_margins = flat_margin[valid_idx]
                _, hardest_idx = torch.topk(-valid_margins, k=keep_count)
                chosen_idx = valid_idx[hardest_idx]
                weight = torch.zeros_like(flat_margin, dtype=dpo_margin.dtype)
                weight[chosen_idx] = 1.0
                return weight.view_as(dpo_margin)

        return None


WEIGHT_REGISTRY = {
    "uniform": dpo_UniformWeight,
    "attr_prior": dpo_AttrPriorWeight,
    "semantic_ref_cache": dpo_SemanticRefCacheWeight,
    "semantic_ref_online": dpo_SemanticRefOnlineWeight,
    "dynamic_margin": dpo_DynamicMarginWeight,
}


def build_weight_strategy(
        atomic_text_weight_mode="uniform",
        attr_weights=None,
        semantic_weight_scale=1.0,
        semantic_weight_min=0.5,
        semantic_weight_max=2.0,
        hard_mining="none",
        hard_temperature=0.1,
        hard_topk=0,
):
    if atomic_text_weight_mode == "uniform":
        return WEIGHT_REGISTRY["uniform"]()
    if atomic_text_weight_mode == "attr_prior":
        return WEIGHT_REGISTRY["attr_prior"](attr_weights=attr_weights)
    if atomic_text_weight_mode == "semantic_ref_cache":
        return WEIGHT_REGISTRY["semantic_ref_cache"]()
    if atomic_text_weight_mode == "semantic_ref_online":
        return WEIGHT_REGISTRY["semantic_ref_online"](
            scale=semantic_weight_scale,
            weight_min=semantic_weight_min,
            weight_max=semantic_weight_max,
        )
    if atomic_text_weight_mode == "dynamic_margin":
        return WEIGHT_REGISTRY["dynamic_margin"](
            hard_mining=hard_mining,
            temperature=hard_temperature,
            hard_topk=hard_topk,
            weight_min=semantic_weight_min,
            weight_max=semantic_weight_max,
        )
    raise ValueError(f"Unsupported atomic_text_weight_mode: {atomic_text_weight_mode}")


class dpo_Global4WayDpoTerm(nn.Module):
    TERM_LOG_KEYS = (
        "global_dpo_pos_image_text_pref_loss",
        "global_dpo_neg_image_text_pref_loss",
        "global_dpo_pos_text_image_pref_loss",
        "global_dpo_neg_text_image_pref_loss",
    )
    TERM_WEIGHT_LOG_KEYS = (
        "global_dpo_pos_image_text_pref_weight",
        "global_dpo_neg_image_text_pref_weight",
        "global_dpo_pos_text_image_pref_weight",
        "global_dpo_neg_text_image_pref_weight",
    )

    def __init__(self, use_4way=True, term_weights=None):
        super().__init__()
        self.use_4way = use_4way
        self.term_weights = tuple(float(weight) for weight in (term_weights or (1.0, 1.0, 1.0, 1.0)))
        if len(self.term_weights) != 4:
            raise ValueError("global 4-way term_weights must contain exactly 4 values")
        if any(weight < 0 for weight in self.term_weights):
            raise ValueError("global 4-way term_weights must be non-negative")
        if sum(self.term_weights) <= 0:
            raise ValueError("global 4-way term_weights must sum to a positive value")

    def forward(self, x: dpo_AtomicDpoInputs, core: dpo_DpoCore):
        s_th_pos1 = core.score(x.image_pos, x.text_pos)
        s_th_pos2 = core.score(x.image_neg, x.text_neg)
        s_th_neg1 = core.score(x.image_pos, x.text_neg)
        s_th_neg2 = core.score(x.image_neg, x.text_pos)

        s_ref_pos1 = core.score(x.ref_image_pos, x.ref_text_pos)
        s_ref_pos2 = core.score(x.ref_image_neg, x.ref_text_neg)
        s_ref_neg1 = core.score(x.ref_image_pos, x.ref_text_neg)
        s_ref_neg2 = core.score(x.ref_image_neg, x.ref_text_pos)

        term_loss_vecs = (
            core.dpo_pair_loss(s_th_pos1, s_th_neg1, s_ref_pos1, s_ref_neg1),
            core.dpo_pair_loss(s_th_pos2, s_th_neg2, s_ref_pos2, s_ref_neg2),
            core.dpo_pair_loss(s_th_pos1, s_th_neg2, s_ref_pos1, s_ref_neg2),
            core.dpo_pair_loss(s_th_pos2, s_th_neg1, s_ref_pos2, s_ref_neg1),
        )
        weight_tensor = x.image_pos.new_tensor(self.term_weights, dtype=s_th_pos1.dtype)
        log_weights = [0.0, 0.0, 0.0, 0.0]

        if self.use_4way:
            # Use the provided weights directly so the caller controls the
            # absolute contribution of each 4-way preference term.
            effective_weights = weight_tensor
            loss_vec = sum(
                effective_weights[idx] * term_loss_vec
                for idx, term_loss_vec in enumerate(term_loss_vecs)
            )
            log_weights = effective_weights.tolist()
        else:
            loss_vec = term_loss_vecs[0]
            log_weights = [1.0, 0.0, 0.0, 0.0]

        loss = loss_vec.mean()
        logs = {"global_dpo_loss": loss}
        for log_key, term_loss_vec in zip(self.TERM_LOG_KEYS, term_loss_vecs):
            logs[log_key] = term_loss_vec.mean()
        for log_key, weight in zip(self.TERM_WEIGHT_LOG_KEYS, log_weights):
            logs[log_key] = x.image_pos.new_tensor(weight, dtype=s_th_pos1.dtype)
        return {"loss": loss, "logs": logs}


class dpo_AtomicTextDpoTerm(nn.Module):
    def __init__(self, weight_strategy=None):
        super().__init__()
        self.weight_strategy = weight_strategy or dpo_UniformWeight()

    def forward(self, x: dpo_AtomicDpoInputs, core: dpo_DpoCore):
        zero = x.image_pos.new_zeros(())
        zero_logs = {
            "atomic_text_dpo_loss": zero,
            "atomic_text_theta_margin": zero,
            "atomic_text_ref_margin": zero,
            "atomic_text_dpo_margin": zero,
            "atomic_text_weight_mean": zero,
            "atomic_text_weight_max": zero,
            "atomic_text_weight_min": zero,
        }
        if (
                x.atomic_text_pos is None
                or x.atomic_text_neg is None
                or x.ref_atomic_text_pos is None
                or x.ref_atomic_text_neg is None
        ):
            return {"loss": zero, "logs": zero_logs}

        s_th_pos = core.score(x.image_pos, x.atomic_text_pos)
        s_th_neg = core.score(x.image_pos, x.atomic_text_neg)
        s_ref_pos = core.score(x.ref_image_pos, x.ref_atomic_text_pos)
        s_ref_neg = core.score(x.ref_image_pos, x.ref_atomic_text_neg)

        loss_mat = core.dpo_pair_loss(s_th_pos, s_th_neg, s_ref_pos, s_ref_neg)
        weight = self.weight_strategy(
            x=x,
            loss_mat=loss_mat,
            s_th_pos=s_th_pos,
            s_th_neg=s_th_neg,
            s_ref_pos=s_ref_pos,
            s_ref_neg=s_ref_neg,
        )
        loss = weighted_reduce(loss_mat, mask=x.active_atomic_mask, weight=weight)

        logged_weight = weight if weight is not None else torch.ones_like(loss_mat)
        weight_mean, weight_max, weight_min = summarize_weight_stats(
            logged_weight,
            mask=x.active_atomic_mask,
        )
        theta_margin = s_th_pos - s_th_neg
        ref_margin = s_ref_pos - s_ref_neg
        dpo_margin = theta_margin - ref_margin

        return {
            "loss": loss,
            "logs": {
                "atomic_text_dpo_loss": loss,
                "atomic_text_theta_margin": weighted_reduce(theta_margin, mask=x.active_atomic_mask),
                "atomic_text_ref_margin": weighted_reduce(ref_margin, mask=x.active_atomic_mask),
                "atomic_text_dpo_margin": weighted_reduce(dpo_margin, mask=x.active_atomic_mask),
                "atomic_text_weight_mean": weight_mean,
                "atomic_text_weight_max": weight_max,
                "atomic_text_weight_min": weight_min,
            },
        }


class dpo_AtomicVisualDpoTerm(nn.Module):
    def __init__(self, neg_source="global_neg"):
        super().__init__()
        self.neg_source = neg_source

    def forward(self, x: dpo_AtomicDpoInputs, core: dpo_DpoCore):
        zero = x.image_pos.new_zeros(())
        zero_logs = {
            "atomic_visual_dpo_loss": zero,
            "atomic_visual_theta_margin": zero,
            "atomic_visual_ref_margin": zero,
            "atomic_visual_dpo_margin": zero,
            "atomic_visual_neg_other_mismatch_mean": zero,
            "atomic_visual_neg_exact_rate": zero,
        }
        if x.atomic_text_pos is None or x.ref_atomic_text_pos is None:
            return {"loss": zero, "logs": zero_logs}

        s_th_pos = core.score(x.image_pos, x.atomic_text_pos)
        s_ref_pos = core.score(x.ref_image_pos, x.ref_atomic_text_pos)
        visual_mask = combine_optional_masks(
            x.active_atomic_mask,
            x.atomic_visual_neg_found_mask,
        )

        if self.neg_source == "atomic_matched":
            if x.atomic_image_neg is None or x.ref_atomic_image_neg is None:
                raise ValueError(
                    "atomic_matched visual negatives require atomic_image_neg and ref_atomic_image_neg features"
                )
            s_th_neg = core.score(x.atomic_image_neg, x.atomic_text_pos)
            s_ref_neg = core.score(x.ref_atomic_image_neg, x.ref_atomic_text_pos)
        else:
            s_th_neg = core.score(x.image_neg, x.atomic_text_pos)
            s_ref_neg = core.score(x.ref_image_neg, x.ref_atomic_text_pos)

        loss_mat = core.dpo_pair_loss(s_th_pos, s_th_neg, s_ref_pos, s_ref_neg)
        loss = weighted_reduce(loss_mat, mask=visual_mask)
        theta_margin = s_th_pos - s_th_neg
        ref_margin = s_ref_pos - s_ref_neg
        dpo_margin = theta_margin - ref_margin
        mismatch_mean = zero
        exact_rate = zero
        if self.neg_source == "atomic_matched" and x.atomic_visual_neg_mismatch_count is not None:
            mismatch_count = x.atomic_visual_neg_mismatch_count.to(
                device=loss_mat.device,
                dtype=loss_mat.dtype,
            )
            mismatch_mean = weighted_reduce(mismatch_count, mask=visual_mask)
            exact_rate = weighted_reduce(
                (mismatch_count == 0).to(dtype=loss_mat.dtype),
                mask=visual_mask,
            )

        return {
            "loss": loss,
            "logs": {
                "atomic_visual_dpo_loss": loss,
                "atomic_visual_theta_margin": weighted_reduce(theta_margin, mask=visual_mask),
                "atomic_visual_ref_margin": weighted_reduce(ref_margin, mask=visual_mask),
                "atomic_visual_dpo_margin": weighted_reduce(dpo_margin, mask=visual_mask),
                "atomic_visual_neg_other_mismatch_mean": mismatch_mean,
                "atomic_visual_neg_exact_rate": exact_rate,
            },
        }


class dpo_Atomic4WayDpoTerm(nn.Module):
    TERM_LOG_KEYS = (
        "atomic_fourway_pos_image_text_pref_loss",
        "atomic_fourway_neg_image_text_pref_loss",
        "atomic_fourway_pos_text_image_pref_loss",
        "atomic_fourway_neg_text_image_pref_loss",
    )
    TERM_WEIGHT_LOG_KEYS = (
        "atomic_fourway_pos_image_text_pref_weight",
        "atomic_fourway_neg_image_text_pref_weight",
        "atomic_fourway_pos_text_image_pref_weight",
        "atomic_fourway_neg_text_image_pref_weight",
    )

    def __init__(self, term_weights=None, neg_source="global_neg"):
        super().__init__()
        self.neg_source = neg_source
        self.term_weights = tuple(float(weight) for weight in (term_weights or (1.0, 1.0, 1.0, 1.0)))
        if len(self.term_weights) != 4:
            raise ValueError("atomic 4-way term_weights must contain exactly 4 values")
        if any(weight < 0 for weight in self.term_weights):
            raise ValueError("atomic 4-way term_weights must be non-negative")
        if sum(self.term_weights) <= 0:
            raise ValueError("atomic 4-way term_weights must sum to a positive value")

    def forward(self, x: dpo_AtomicDpoInputs, core: dpo_DpoCore):
        zero = x.image_pos.new_zeros(())
        zero_logs = {
            "atomic_fourway_dpo_loss": zero,
            "atomic_fourway_theta_margin": zero,
            "atomic_fourway_ref_margin": zero,
            "atomic_fourway_dpo_margin": zero,
            "atomic_fourway_neg_other_mismatch_mean": zero,
            "atomic_fourway_neg_exact_rate": zero,
        }
        for log_key in self.TERM_LOG_KEYS + self.TERM_WEIGHT_LOG_KEYS:
            zero_logs[log_key] = zero

        if (
                x.atomic_text_pos is None
                or x.atomic_text_neg is None
                or x.ref_atomic_text_pos is None
                or x.ref_atomic_text_neg is None
        ):
            return {"loss": zero, "logs": zero_logs}

        active_mask = x.active_atomic_mask
        mismatch_mean = zero
        exact_rate = zero

        if self.neg_source == "atomic_matched":
            if x.atomic_image_neg is None or x.ref_atomic_image_neg is None:
                raise ValueError(
                    "atomic_matched atomic 4-way negatives require atomic_image_neg and ref_atomic_image_neg features"
                )
            image_neg = x.atomic_image_neg
            ref_image_neg = x.ref_atomic_image_neg
            active_mask = combine_optional_masks(active_mask, x.atomic_visual_neg_found_mask)
            if x.atomic_visual_neg_mismatch_count is not None:
                mismatch_count = x.atomic_visual_neg_mismatch_count.to(
                    device=x.image_pos.device,
                    dtype=x.image_pos.dtype,
                )
                mismatch_mean = weighted_reduce(mismatch_count, mask=active_mask)
                exact_rate = weighted_reduce(
                    (mismatch_count == 0).to(dtype=x.image_pos.dtype),
                    mask=active_mask,
                )
        else:
            image_neg = x.image_neg
            ref_image_neg = x.ref_image_neg

        s_th_pos1 = core.score(x.image_pos, x.atomic_text_pos)
        s_th_pos2 = core.score(image_neg, x.atomic_text_neg)
        s_th_neg1 = core.score(x.image_pos, x.atomic_text_neg)
        s_th_neg2 = core.score(image_neg, x.atomic_text_pos)

        s_ref_pos1 = core.score(x.ref_image_pos, x.ref_atomic_text_pos)
        s_ref_pos2 = core.score(ref_image_neg, x.ref_atomic_text_neg)
        s_ref_neg1 = core.score(x.ref_image_pos, x.ref_atomic_text_neg)
        s_ref_neg2 = core.score(ref_image_neg, x.ref_atomic_text_pos)

        term_loss_mats = (
            core.dpo_pair_loss(s_th_pos1, s_th_neg1, s_ref_pos1, s_ref_neg1),
            core.dpo_pair_loss(s_th_pos2, s_th_neg2, s_ref_pos2, s_ref_neg2),
            core.dpo_pair_loss(s_th_pos1, s_th_neg2, s_ref_pos1, s_ref_neg2),
            core.dpo_pair_loss(s_th_pos2, s_th_neg1, s_ref_pos2, s_ref_neg1),
        )
        term_theta_margins = (
            s_th_pos1 - s_th_neg1,
            s_th_pos2 - s_th_neg2,
            s_th_pos1 - s_th_neg2,
            s_th_pos2 - s_th_neg1,
        )
        term_ref_margins = (
            s_ref_pos1 - s_ref_neg1,
            s_ref_pos2 - s_ref_neg2,
            s_ref_pos1 - s_ref_neg2,
            s_ref_pos2 - s_ref_neg1,
        )

        weight_tensor = x.image_pos.new_tensor(self.term_weights, dtype=s_th_pos1.dtype)
        effective_weights = weight_tensor
        loss_mat = sum(
            effective_weights[idx] * term_loss_mat
            for idx, term_loss_mat in enumerate(term_loss_mats)
        )
        loss = weighted_reduce(loss_mat, mask=active_mask)

        theta_margin = sum(
            effective_weights[idx] * term_theta_margin
            for idx, term_theta_margin in enumerate(term_theta_margins)
        )
        ref_margin = sum(
            effective_weights[idx] * term_ref_margin
            for idx, term_ref_margin in enumerate(term_ref_margins)
        )
        dpo_margin = theta_margin - ref_margin

        logs = {
            "atomic_fourway_dpo_loss": loss,
            "atomic_fourway_theta_margin": weighted_reduce(theta_margin, mask=active_mask),
            "atomic_fourway_ref_margin": weighted_reduce(ref_margin, mask=active_mask),
            "atomic_fourway_dpo_margin": weighted_reduce(dpo_margin, mask=active_mask),
            "atomic_fourway_neg_other_mismatch_mean": mismatch_mean,
            "atomic_fourway_neg_exact_rate": exact_rate,
        }
        for log_key, term_loss_mat in zip(self.TERM_LOG_KEYS, term_loss_mats):
            logs[log_key] = weighted_reduce(term_loss_mat, mask=active_mask)
        for log_key, weight in zip(self.TERM_WEIGHT_LOG_KEYS, effective_weights.tolist()):
            logs[log_key] = x.image_pos.new_tensor(weight, dtype=s_th_pos1.dtype)
        return {"loss": loss, "logs": logs}


TERM_REGISTRY = {
    "global": dpo_Global4WayDpoTerm,
    "atomic_fourway": dpo_Atomic4WayDpoTerm,
    "atomic_text": dpo_AtomicTextDpoTerm,
    "atomic_visual": dpo_AtomicVisualDpoTerm,
}


class dpo_CompositeAtomicDpoLoss(nn.Module):
    def __init__(
            self,
            core: dpo_DpoCore,
            terms,
            atomic_train_mode="row_all",
            num_atomic_attrs=6,
            scale_global_in_prop_sample=False,
            atomic_dropout_k=0,
            atomic_dropout_p=0.0,
    ):
        super().__init__()
        self.core = core
        self.atomic_train_mode = atomic_train_mode
        self.num_atomic_attrs = num_atomic_attrs
        self.scale_global_in_prop_sample = scale_global_in_prop_sample
        self.atomic_dropout_k = atomic_dropout_k
        self.atomic_dropout_p = atomic_dropout_p
        self.term_names = [name for name, _, _ in terms]
        self.term_weights = [weight for _, weight, _ in terms]
        self.terms = nn.ModuleList([term for _, _, term in terms])

    def _effective_global_weight(self, base_weight):
        if self.atomic_train_mode == "prop_sample" and self.scale_global_in_prop_sample:
            return base_weight / float(max(self.num_atomic_attrs, 1))
        return base_weight

    def _empty_logs(self, x: dpo_AtomicDpoInputs):
        zero = x.image_pos.new_zeros(())
        return {
            "lambda_global_eff": zero,
            "global_dpo_loss": zero,
            "global_dpo_pos_image_text_pref_loss": zero,
            "global_dpo_neg_image_text_pref_loss": zero,
            "global_dpo_pos_text_image_pref_loss": zero,
            "global_dpo_neg_text_image_pref_loss": zero,
            "global_dpo_pos_image_text_pref_weight": zero,
            "global_dpo_neg_image_text_pref_weight": zero,
            "global_dpo_pos_text_image_pref_weight": zero,
            "global_dpo_neg_text_image_pref_weight": zero,
            "atomic_fourway_dpo_loss": zero,
            "atomic_fourway_theta_margin": zero,
            "atomic_fourway_ref_margin": zero,
            "atomic_fourway_dpo_margin": zero,
            "atomic_fourway_neg_other_mismatch_mean": zero,
            "atomic_fourway_neg_exact_rate": zero,
            "atomic_fourway_pos_image_text_pref_loss": zero,
            "atomic_fourway_neg_image_text_pref_loss": zero,
            "atomic_fourway_pos_text_image_pref_loss": zero,
            "atomic_fourway_neg_text_image_pref_loss": zero,
            "atomic_fourway_pos_image_text_pref_weight": zero,
            "atomic_fourway_neg_image_text_pref_weight": zero,
            "atomic_fourway_pos_text_image_pref_weight": zero,
            "atomic_fourway_neg_text_image_pref_weight": zero,
            "atomic_text_dpo_loss": zero,
            "atomic_text_theta_margin": zero,
            "atomic_text_ref_margin": zero,
            "atomic_text_dpo_margin": zero,
            "atomic_text_weight_mean": zero,
            "atomic_text_weight_max": zero,
            "atomic_text_weight_min": zero,
            "atomic_visual_dpo_loss": zero,
            "atomic_visual_theta_margin": zero,
            "atomic_visual_ref_margin": zero,
            "atomic_visual_dpo_margin": zero,
            "atomic_visual_neg_other_mismatch_mean": zero,
            "atomic_visual_neg_exact_rate": zero,
        }

    def forward(self, x: dpo_AtomicDpoInputs, output_dict=False):
        if x.atomic_mask is not None and x.atomic_mask.dim() == 2:
            x.effective_atomic_mask = apply_atomic_dropout_mask(
                x.atomic_mask,
                training=self.training,
                atomic_dropout_k=self.atomic_dropout_k,
                atomic_dropout_p=self.atomic_dropout_p,
            )
        else:
            x.effective_atomic_mask = x.atomic_mask

        total_loss = x.image_pos.new_zeros(())
        logs = self._empty_logs(x)

        for name, base_weight, term in zip(self.term_names, self.term_weights, self.terms):
            term_weight = base_weight
            if name == "global":
                term_weight = self._effective_global_weight(base_weight)
                logs["lambda_global_eff"] = x.image_pos.new_tensor(term_weight)

            out = term(x, self.core)
            total_loss = total_loss + term_weight * out["loss"]
            logs.update(out["logs"])
            logs[f"lambda_{name}"] = x.image_pos.new_tensor(term_weight)

        logs["dpo_loss"] = total_loss
        return logs if output_dict else total_loss


def build_atomic_dpo_loss_impl(
        alpha=1.0,
        beta=1.0,
        normalize_features=True,
        lambda_global=1.0,
        global_fourway_weights=None,
        lambda_atomic_fourway=0.0,
        atomic_fourway_weights=None,
        lambda_atomic_text=1.0,
        lambda_atomic_visual=0.5,
        use_global_4way=True,
        use_neg_image_for_atomic_visual=True,
        atomic_visual_neg_source="global_neg",
        atomic_train_mode="row_all",
        num_atomic_attrs=6,
        scale_global_in_prop_sample=False,
        atomic_dropout_k=0,
        atomic_dropout_p=0.0,
        attr_weights=None,
        atomic_text_weight_mode="uniform",
        semantic_weight_scale=1.0,
        semantic_weight_min=0.5,
        semantic_weight_max=2.0,
        hard_mining="none",
        hard_temperature=0.1,
        hard_topk=0,
):
    core = dpo_DpoCore(
        alpha=alpha,
        beta=beta,
        normalize_features=normalize_features,
    )
    atomic_text_weight = build_weight_strategy(
        atomic_text_weight_mode=atomic_text_weight_mode,
        attr_weights=attr_weights,
        semantic_weight_scale=semantic_weight_scale,
        semantic_weight_min=semantic_weight_min,
        semantic_weight_max=semantic_weight_max,
        hard_mining=hard_mining,
        hard_temperature=hard_temperature,
        hard_topk=hard_topk,
    )

    terms = []
    if lambda_global > 0:
        terms.append((
            "global",
            lambda_global,
            TERM_REGISTRY["global"](
                use_4way=use_global_4way,
                term_weights=global_fourway_weights,
            ),
        ))
    if lambda_atomic_fourway > 0:
        terms.append((
            "atomic_fourway",
            lambda_atomic_fourway,
            TERM_REGISTRY["atomic_fourway"](
                term_weights=atomic_fourway_weights,
                neg_source=atomic_visual_neg_source,
            ),
        ))
    if lambda_atomic_text > 0:
        terms.append((
            "atomic_text",
            lambda_atomic_text,
            TERM_REGISTRY["atomic_text"](weight_strategy=atomic_text_weight),
        ))
    if lambda_atomic_visual > 0 and use_neg_image_for_atomic_visual:
        terms.append((
            "atomic_visual",
            lambda_atomic_visual,
            TERM_REGISTRY["atomic_visual"](neg_source=atomic_visual_neg_source),
        ))

    return dpo_CompositeAtomicDpoLoss(
        core=core,
        terms=terms,
        atomic_train_mode=atomic_train_mode,
        num_atomic_attrs=num_atomic_attrs,
        scale_global_in_prop_sample=scale_global_in_prop_sample,
        atomic_dropout_k=atomic_dropout_k,
        atomic_dropout_p=atomic_dropout_p,
    )


def resolve_atomic_dpo_config(
        atomic_loss_preset="full",
        alpha=1.0,
        beta=1.0,
        normalize_features=True,
        lambda_global=None,
        global_fourway_weights=None,
        lambda_atomic_fourway=None,
        atomic_fourway_weights=None,
        lambda_atomic_text=None,
        lambda_atomic_visual=None,
        use_global_4way=True,
        use_neg_image_for_atomic_visual=None,
        atomic_visual_neg_source=None,
        atomic_train_mode=None,
        num_atomic_attrs=6,
        scale_global_in_prop_sample=None,
        atomic_dropout_k=None,
        atomic_dropout_p=None,
        attr_weights=None,
        atomic_text_weight_mode=None,
        semantic_weight_scale=1.0,
        semantic_weight_min=0.5,
        semantic_weight_max=2.0,
        hard_mining="none",
        hard_temperature=0.1,
        hard_topk=0,
):
    preset_cfg = get_atomic_loss_preset(atomic_loss_preset or "full")
    return {
        "alpha": alpha,
        "beta": beta,
        "normalize_features": normalize_features,
        "lambda_global": preset_cfg["lambda_global"] if lambda_global is None else lambda_global,
        "global_fourway_weights": (
            list(preset_cfg["global_fourway_weights"])
            if global_fourway_weights is None else list(global_fourway_weights)
        ),
        "lambda_atomic_fourway": (
            preset_cfg.get("lambda_atomic_fourway", 0.0)
            if lambda_atomic_fourway is None else lambda_atomic_fourway
        ),
        "atomic_fourway_weights": (
            list(preset_cfg.get("atomic_fourway_weights", [1.0, 1.0, 1.0, 1.0]))
            if atomic_fourway_weights is None else list(atomic_fourway_weights)
        ),
        "lambda_atomic_text": preset_cfg["lambda_atomic_text"] if lambda_atomic_text is None else lambda_atomic_text,
        "lambda_atomic_visual": preset_cfg["lambda_atomic_visual"] if lambda_atomic_visual is None else lambda_atomic_visual,
        "use_global_4way": use_global_4way,
        "use_neg_image_for_atomic_visual": (
            preset_cfg["atomic_visual_use_neg_image"]
            if use_neg_image_for_atomic_visual is None else use_neg_image_for_atomic_visual
        ),
        "atomic_visual_neg_source": (
            preset_cfg["atomic_visual_neg_source"]
            if atomic_visual_neg_source is None else atomic_visual_neg_source
        ),
        "atomic_train_mode": preset_cfg["atomic_train_mode"] if atomic_train_mode is None else atomic_train_mode,
        "num_atomic_attrs": num_atomic_attrs,
        "scale_global_in_prop_sample": (
            preset_cfg["scale_global_in_prop_sample"]
            if scale_global_in_prop_sample is None else scale_global_in_prop_sample
        ),
        "atomic_dropout_k": preset_cfg["atomic_dropout_k"] if atomic_dropout_k is None else atomic_dropout_k,
        "atomic_dropout_p": preset_cfg["atomic_dropout_p"] if atomic_dropout_p is None else atomic_dropout_p,
        "attr_weights": attr_weights,
        "atomic_text_weight_mode": (
            preset_cfg["atomic_text_weight_mode"]
            if atomic_text_weight_mode is None else atomic_text_weight_mode
        ),
        "semantic_weight_scale": semantic_weight_scale,
        "semantic_weight_min": semantic_weight_min,
        "semantic_weight_max": semantic_weight_max,
        "hard_mining": hard_mining,
        "hard_temperature": hard_temperature,
        "hard_topk": hard_topk,
    }


class dpo_AtomicDpoLoss(nn.Module):
    def __init__(
            self,
            alpha: float = 1.0,
            beta: float = 1.0,
            normalize_features: bool = True,
            lambda_global: Optional[float] = None,
            global_fourway_weights=None,
            lambda_atomic_fourway: Optional[float] = None,
            atomic_fourway_weights=None,
            lambda_atomic_text: Optional[float] = None,
            lambda_atomic_visual: Optional[float] = None,
            use_global_4way: bool = True,
            use_neg_image_for_atomic_visual: Optional[bool] = None,
            atomic_visual_neg_source: Optional[str] = None,
            atomic_train_mode: Optional[str] = None,
            num_atomic_attrs: int = 6,
            scale_global_in_prop_sample: Optional[bool] = None,
            atomic_dropout_k: Optional[int] = None,
            atomic_dropout_p: Optional[float] = None,
            attr_weights=None,
            atomic_text_weight_mode: Optional[str] = None,
            semantic_weight_scale: float = 1.0,
            semantic_weight_min: float = 0.5,
            semantic_weight_max: float = 2.0,
            hard_mining: str = "none",
            hard_temperature: float = 0.1,
            hard_topk: int = 0,
            atomic_loss_preset: str = "full",
    ):
        super().__init__()
        self.atomic_loss_preset = atomic_loss_preset
        self.resolved_config = resolve_atomic_dpo_config(
            atomic_loss_preset=atomic_loss_preset,
            alpha=alpha,
            beta=beta,
            normalize_features=normalize_features,
            lambda_global=lambda_global,
            global_fourway_weights=global_fourway_weights,
            lambda_atomic_fourway=lambda_atomic_fourway,
            atomic_fourway_weights=atomic_fourway_weights,
            lambda_atomic_text=lambda_atomic_text,
            lambda_atomic_visual=lambda_atomic_visual,
            use_global_4way=use_global_4way,
            use_neg_image_for_atomic_visual=use_neg_image_for_atomic_visual,
            atomic_visual_neg_source=atomic_visual_neg_source,
            atomic_train_mode=atomic_train_mode,
            num_atomic_attrs=num_atomic_attrs,
            scale_global_in_prop_sample=scale_global_in_prop_sample,
            atomic_dropout_k=atomic_dropout_k,
            atomic_dropout_p=atomic_dropout_p,
            attr_weights=attr_weights,
            atomic_text_weight_mode=atomic_text_weight_mode,
            semantic_weight_scale=semantic_weight_scale,
            semantic_weight_min=semantic_weight_min,
            semantic_weight_max=semantic_weight_max,
            hard_mining=hard_mining,
            hard_temperature=hard_temperature,
            hard_topk=hard_topk,
        )
        self.impl = build_atomic_dpo_loss_impl(**self.resolved_config)

    def forward(
            self,
            image_features_pos,
            image_features_neg,
            text_features_pos,
            text_features_neg,
            ref_image_features_pos,
            ref_image_features_neg,
            ref_text_features_pos,
            ref_text_features_neg,
            atomic_text_features_pos=None,
            atomic_text_features_neg=None,
            ref_atomic_text_features_pos=None,
            ref_atomic_text_features_neg=None,
            atomic_image_features_neg=None,
            ref_atomic_image_features_neg=None,
            atomic_mask=None,
            atomic_text_weight=None,
            atomic_attr_idx=None,
            atomic_visual_neg_found_mask=None,
            atomic_visual_neg_mismatch_count=None,
            output_dict: bool = False,
    ):
        inputs = dpo_AtomicDpoInputs(
            image_pos=image_features_pos,
            image_neg=image_features_neg,
            text_pos=text_features_pos,
            text_neg=text_features_neg,
            ref_image_pos=ref_image_features_pos,
            ref_image_neg=ref_image_features_neg,
            ref_text_pos=ref_text_features_pos,
            ref_text_neg=ref_text_features_neg,
            atomic_text_pos=atomic_text_features_pos,
            atomic_text_neg=atomic_text_features_neg,
            ref_atomic_text_pos=ref_atomic_text_features_pos,
            ref_atomic_text_neg=ref_atomic_text_features_neg,
            atomic_image_neg=atomic_image_features_neg,
            ref_atomic_image_neg=ref_atomic_image_features_neg,
            atomic_mask=atomic_mask,
            atomic_text_weight=atomic_text_weight,
            atomic_attr_idx=atomic_attr_idx,
            atomic_visual_neg_found_mask=atomic_visual_neg_found_mask,
            atomic_visual_neg_mismatch_count=atomic_visual_neg_mismatch_count,
        )
        return self.impl(inputs, output_dict=output_dict)


DpoLoss = dpo_DpoLoss
AtomicDpoInputs = dpo_AtomicDpoInputs
DpoCore = dpo_DpoCore
UniformWeight = dpo_UniformWeight
AttrPriorWeight = dpo_AttrPriorWeight
SemanticRefCacheWeight = dpo_SemanticRefCacheWeight
SemanticRefOnlineWeight = dpo_SemanticRefOnlineWeight
DynamicMarginWeight = dpo_DynamicMarginWeight
Global4WayDpoTerm = dpo_Global4WayDpoTerm
Atomic4WayDpoTerm = dpo_Atomic4WayDpoTerm
AtomicTextDpoTerm = dpo_AtomicTextDpoTerm
AtomicVisualDpoTerm = dpo_AtomicVisualDpoTerm
CompositeAtomicDpoLoss = dpo_CompositeAtomicDpoLoss
AtomicDpoLoss = dpo_AtomicDpoLoss

# atomic_loss.txt search aliases
DpoGlobal4WayTerm = dpo_Global4WayDpoTerm
DpoAtomic4WayTerm = dpo_Atomic4WayDpoTerm
DpoAtomicTextTerm = dpo_AtomicTextDpoTerm
DpoAtomicVisualTerm = dpo_AtomicVisualDpoTerm
DpoUniformWeight = dpo_UniformWeight
DpoAttrPriorWeight = dpo_AttrPriorWeight
DpoSemanticRefOnlineWeight = dpo_SemanticRefOnlineWeight
DpoSemanticRefCacheWeight = dpo_SemanticRefCacheWeight
DpoDynamicMarginWeight = dpo_DynamicMarginWeight
DpoCompositeLoss = dpo_CompositeAtomicDpoLoss
DpoAtomicLoss = dpo_AtomicDpoLoss

class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss
