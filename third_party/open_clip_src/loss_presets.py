from copy import deepcopy


def _preset(name, **overrides):
    base = {
        "name": name,
        "dpo_loss_type": "4way",
        "lambda_global": 1.0,
        "global_fourway_weights": [1.0, 1.0, 1.0, 1.0],
        "lambda_atomic_fourway": 0.0,
        "atomic_fourway_weights": [1.0, 1.0, 1.0, 1.0],
        "lambda_atomic_text": 0.0,
        "lambda_atomic_visual": 0.0,
        "atomic_train_mode": "row_all",
        "atomic_text_weight_mode": "uniform",
        "atomic_visual_use_neg_image": True,
        "atomic_visual_neg_source": "global_neg",
        "atomic_visual_match_pool": "train_csv",
        "atomic_visual_max_other_mismatches": 4,
        "scale_global_in_prop_sample": False,
        "atomic_dropout_k": 0,
        "atomic_dropout_p": 0.0,
        "attr_loss_weights": None,
        "hard_mining": "none",
        "hard_temperature": 0.1,
        "hard_topk": 0,
        "semantic_weight_scale": 1.0,
        "semantic_weight_min": 0.5,
        "semantic_weight_max": 2.0,
    }
    base.update(overrides)
    return base


ATOMIC_LOSS_PRESETS = {
    "loss_001": _preset(
        "globaltext+globalvisual",
    ),
    "loss_002": _preset(
        "globaltext+globalvisual+atomictext(row_all, uniform)",
        lambda_atomic_text=1.0,
    ),
    "loss_003": _preset(
        "globaltext+globalvisual+atomictext(prop_sample, uniform, global_scale_off)",
        lambda_atomic_text=1.0,
        atomic_train_mode="prop_sample",
        atomic_visual_use_neg_image=False,
        scale_global_in_prop_sample=False,
    ),
    "loss_004": _preset(
        "atomictext(prop_sample, uniform)",
        lambda_global=0.0,
        lambda_atomic_text=1.0,
        atomic_train_mode="prop_sample",
        atomic_visual_use_neg_image=False,
    ),
    "loss_005": _preset(
        "globaltext+globalvisual+atomicvisual",
        lambda_atomic_visual=0.5,
    ),
    "loss_006": _preset(
        "globaltext+globalvisual+atomictext+atomicvisual(row_all, uniform)",
        lambda_atomic_text=1.0,
        lambda_atomic_visual=0.5,
    ),
    "loss_007": _preset(
        "globaltext+globalvisual+atomictext+atomicvisual(row_all, uniform, dropout_k1)",
        lambda_atomic_text=1.0,
        lambda_atomic_visual=0.5,
        atomic_dropout_k=1,
    ),
    "loss_008": _preset(
        "globaltext+globalvisual+atomictext+atomicvisual(prop_sample, uniform)",
        lambda_atomic_text=1.0,
        lambda_atomic_visual=0.5,
        atomic_train_mode="prop_sample",
        scale_global_in_prop_sample=True,
    ),
    "loss_009": _preset(
        "globaltext+globalvisual+weighted_atomictext(row_all, attr_prior)",
        lambda_atomic_text=1.0,
        atomic_text_weight_mode="attr_prior",
        atomic_visual_use_neg_image=False,
        attr_loss_weights=[1.0, 1.0, 1.2, 1.2, 1.3, 1.3],
    ),
    "loss_010": _preset(
        "globaltext+globalvisual+weighted_atomictext(row_all, semantic_online)",
        lambda_global=0.5,
        lambda_atomic_text=1.5,
        atomic_text_weight_mode="semantic_ref_online",
        atomic_visual_use_neg_image=False,
    ),
    "loss_011": _preset(
        "globaltext+globalvisual+weighted_atomictext(row_all, semantic_cache)",
        lambda_global=0.5,
        lambda_atomic_text=1.5,
        atomic_text_weight_mode="semantic_ref_cache",
        atomic_visual_use_neg_image=False,
    ),
    "loss_012": _preset(
        "globaltext+globalvisual+weighted_atomictext(row_all, dynamic_margin_weight)",
        lambda_atomic_text=1.0,
        atomic_text_weight_mode="dynamic_margin",
        atomic_visual_use_neg_image=False,
        hard_mining="margin_weight",
    ),
    "loss_013": _preset(
        "globaltext+globalvisual+atomictext+atomicvisual(row_all, uniform, atomic_neg_image)",
        lambda_atomic_text=1.0,
        lambda_atomic_visual=0.5,
        atomic_visual_neg_source="atomic_matched",
        atomic_visual_match_pool="all_metadata",
        atomic_visual_max_other_mismatches=4,
    ),
    "loss_014": _preset(
        "atomic4way(row_all, uniform, atomic_neg_image)",
        lambda_global=0.0,
        lambda_atomic_fourway=1.0,
        atomic_visual_neg_source="atomic_matched",
        atomic_visual_match_pool="all_metadata",
        atomic_visual_max_other_mismatches=4,
    ),
}


ATOMIC_LOSS_PRESET_ALIASES = {
    "global_only": "loss_001",
    "global_atomic_text": "loss_002",
    "prop_text_with_global_1.0": "loss_003",
    "prop_text_only": "loss_004",
    "global_atomic_visual": "loss_005",
    "full": "loss_006",
    "row_dropout_k1": "loss_007",
    "prop_full": "loss_008",
    "semantic_online": "loss_010",
    "semantic_precompute": "loss_011",
    "sb_at_dpo": "loss_010",
    "full_atomic_matched": "loss_013",
    "atomic_fourway_only": "loss_014",
}


ATOMIC_LOSS_PRESET_CHOICES = tuple(ATOMIC_LOSS_PRESETS.keys())


def resolve_atomic_loss_preset_name(name):
    return ATOMIC_LOSS_PRESET_ALIASES.get(name, name)


def get_atomic_loss_preset(name):
    resolved_name = resolve_atomic_loss_preset_name(name)
    if resolved_name not in ATOMIC_LOSS_PRESETS:
        raise KeyError(f"Unknown atomic loss preset: {name}")
    return deepcopy(ATOMIC_LOSS_PRESETS[resolved_name])
