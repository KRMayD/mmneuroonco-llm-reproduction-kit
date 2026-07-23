import argparse
import ast
import sys

from loss_presets import (
    ATOMIC_LOSS_PRESET_CHOICES,
    get_atomic_loss_preset,
    resolve_atomic_loss_preset_name,
)


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def _flag_was_passed(cli_args, flag):
    if not cli_args:
        return False
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in cli_args)


def _parse_fourway_weights(flag_name, value):
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"{flag_name} must contain exactly 4 comma-separated floats"
        )

    try:
        weights = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{flag_name} must contain valid floats"
        ) from exc

    if any(weight < 0 for weight in weights):
        raise argparse.ArgumentTypeError(
            f"{flag_name} must be non-negative"
        )
    if sum(weights) <= 0:
        raise argparse.ArgumentTypeError(
            f"{flag_name} must sum to a positive value"
        )
    return weights


def _parse_global_fourway_weights(value):
    return _parse_fourway_weights("--global-fourway-weights", value)


def _parse_atomic_fourway_weights(value):
    return _parse_fourway_weights("--atomic-fourway-weights", value)


def parse_args(args):
    cli_args = list(args) if args is not None else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )
    
    # DPO Arguments
    parser.add_argument(
        "--dpo-loss",
        default=False,
        action="store_true",
        help="Whether to use DPO loss."
    )
    parser.add_argument(
        "--beta-dpo",
        type=float,
        default=0.1,
        help="Beta parameter for DPO loss."
    )
    parser.add_argument(
        "--dpo-loss-type",
        type=str,
        default="4way",
        choices=["4way", "text", "image"],
        help="Which version of DPO loss to use."
    )
    parser.add_argument(
        "--dpo-mode",
        type=str,
        default="standard",
        choices=["standard", "atomic"],
        help="Which DPO data/loss pipeline to use."
    )
    parser.add_argument(
        "--disable-feature-normalization",
        default=False,
        action="store_true",
        help=(
            "Use raw image/text embeddings and dot-product scores for DPO. "
            "This is an ablation; the default uses normalized cosine similarities."
        ),
    )
    parser.add_argument(
        "--atomic-attrs",
        type=str,
        default="location,size,shape,orientation,boundary,disease_type",
        help="Comma-separated atomic attribute names used to compose atomic_<attr>_{pos,neg} columns."
    )
    parser.add_argument(
        "--atomic-pos-suffix",
        type=str,
        default="_pos",
        help="Suffix appended to atomic_<attr> for positive captions."
    )
    parser.add_argument(
        "--atomic-neg-suffix",
        type=str,
        default="_neg",
        help="Suffix appended to atomic_<attr> for negative captions."
    )
    parser.add_argument(
        "--atomic-loss-preset",
        type=resolve_atomic_loss_preset_name,
        default="loss_006",
        choices=list(ATOMIC_LOSS_PRESET_CHOICES),
        help="Canonical atomic loss preset ID. See atomic_loss_short.txt and loss_presets.py for exact definitions."
    )
    parser.add_argument(
        "--lambda-global",
        type=float,
        default=None,
        help="Weight for the global 4-way DPO loss."
    )
    parser.add_argument(
        "--global-fourway-weights",
        type=_parse_global_fourway_weights,
        default=None,
        help=(
            "Comma-separated relative weights for the 4 global DPO subterms in order: "
            "(I+,T+)>(I+,T-), (I-,T-)>(I-,T+), (I+,T+)>(I-,T+), (I-,T-)>(I+,T-). "
            "They are normalized inside the global term so lambda_global stays comparable."
        )
    )
    parser.add_argument(
        "--lambda-atomic-fourway",
        type=float,
        default=None,
        help="Weight for the atomic 4-way DPO loss that uses atomic texts and atomic-matched negative images."
    )
    parser.add_argument(
        "--atomic-fourway-weights",
        type=_parse_atomic_fourway_weights,
        default=None,
        help=(
            "Comma-separated relative weights for the 4 atomic DPO subterms in order: "
            "(I+,t_a+)>(I+,t_a-), (I_a-,t_a-)>(I_a-,t_a+), (I+,t_a+)>(I_a-,t_a+), (I_a-,t_a-)>(I+,t_a-). "
            "They are normalized inside the atomic four-way term so lambda_atomic_fourway stays comparable."
        )
    )
    parser.add_argument(
        "--lambda-atomic-text",
        type=float,
        default=None,
        help="Weight for the atomic text-side DPO loss."
    )
    parser.add_argument(
        "--lambda-atomic-visual",
        type=float,
        default=None,
        help="Weight for the atomic visual grounding DPO loss."
    )
    parser.add_argument(
        "--atomic-visual-use-neg-image",
        type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
        default=True,
        help="Whether atomic visual loss should compare the positive image against filename_neg."
    )
    parser.add_argument(
        "--atomic-visual-neg-source",
        type=str,
        default="global_neg",
        choices=["global_neg", "atomic_matched"],
        help="Negative image source used by atomic image-based branches."
    )
    parser.add_argument(
        "--atomic-visual-attr-json-path",
        type=str,
        default=None,
        help="Metadata JSON used to find attribute-matched atomic visual negatives."
    )
    parser.add_argument(
        "--atomic-visual-match-pool",
        type=str,
        default="train_csv",
        choices=["train_csv", "all_metadata"],
        help="Candidate pool used when searching attribute-matched atomic visual negatives."
    )
    parser.add_argument(
        "--atomic-visual-max-other-mismatches",
        type=int,
        default=4,
        help="Maximum number of non-target atomic mismatches allowed for matched visual negatives."
    )
    parser.add_argument(
        "--atomic-train-mode",
        type=str,
        default="row_all",
        choices=["row_all", "prop_sample"],
        help="Whether to train with all atomic attributes per row or expand each proposition as its own sample."
    )
    parser.add_argument(
        "--scale-global-in-prop-sample",
        default=False,
        action="store_true",
        help="Scale the global loss by 1 / num_atomic_attrs when using prop_sample mode."
    )
    parser.add_argument(
        "--num-atomic-attrs",
        type=int,
        default=None,
        help="Explicit atomic attribute count used for scaling global loss in prop_sample mode."
    )
    parser.add_argument(
        "--atomic-dropout-k",
        type=int,
        default=0,
        help="For row_all mode, sample exactly k valid atomic attributes per row. Set 0 to disable."
    )
    parser.add_argument(
        "--atomic-dropout-p",
        type=float,
        default=0.0,
        help="For row_all mode, randomly drop atomic attributes with probability p. Set 0 to disable."
    )
    parser.add_argument(
        "--attr-loss-weights",
        type=str,
        default=None,
        help="Comma-separated per-attribute loss weights aligned with --atomic-attrs."
    )
    parser.add_argument(
        "--hard-mining",
        type=str,
        default="none",
        choices=["none", "margin_weight", "topk"],
        help="Optional hard atomic proposition weighting strategy."
    )
    parser.add_argument(
        "--hard-temperature",
        type=float,
        default=0.1,
        help="Temperature used by margin-based hard mining."
    )
    parser.add_argument(
        "--hard-topk",
        type=int,
        default=0,
        help="Number of hardest atomic propositions to keep when --hard-mining topk is enabled."
    )
    parser.add_argument(
        "--atomic-text-weight-mode",
        type=str,
        default="uniform",
        choices=["uniform", "attr_prior", "semantic_ref_cache", "semantic_ref_online", "dynamic_margin"],
        help="Difficulty-aware weighting mode applied to the atomic text branch."
    )
    parser.add_argument(
        "--semantic-weight-cache-path",
        type=str,
        default=None,
        help="Optional JSON cache with row-aligned semantic weights for atomic text pairs."
    )
    parser.add_argument(
        "--semantic-weight-scale",
        type=float,
        default=1.0,
        help="Eta scale for semantic similarity text weighting."
    )
    parser.add_argument(
        "--semantic-weight-min",
        type=float,
        default=0.5,
        help="Minimum clamp value for semantic text weights."
    )
    parser.add_argument(
        "--semantic-weight-max",
        type=float,
        default=2.0,
        help="Maximum clamp value for semantic text weights."
    )

    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--csv-img-neg-key",
        type=str,
        default="filename_neg",
        help="For DPO csv-like datasets, the name of the key for the negative images."
    )
    parser.add_argument(
        "--csv-caption-neg-key",
        type=str,
        default="Caption_neg",
        help="For DPO csv-like datasets, the name of the key for the negative captions."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override system default cache path for model & tokenizer file downloads.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown", type=int, default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end", type=float, default=0.0,
        help="End learning rate for cooldown schedule. Default: 0"
    )
    parser.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        '--image-interpolation',
        default=None, type=str, choices=['bicubic', 'bilinear', 'random'],
        help="Override default image resize interpolation"
    )
    parser.add_argument(
        '--image-resize-mode',
        default=None, type=str, choices=['shortest', 'longest', 'squash'],
        help="Override default image resize (& crop) mode during inference"
    )
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        '--force-image-size', type=int, nargs='+', default=None,
        help='Override default image size'
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action='store_true',
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Accelerator to use."
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend",
        default=None,
        type=str,
        help="distributed backend. \"nccl\" for GPU, \"hccl\" for Ascend NPU"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze LayerNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa."
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa."
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one."
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help='Which model arch to distill from, if any.'
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help='Which pre-trained weights to distill from, if any.'
    )
    parser.add_argument(
        "--dpo-ref-checkpoint",
        default=None,
        type=str,
        help='Path to the reference model checkpoint for DPO training.'
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help='Replace the network linear layers from the bitsandbytes library. '
        'Allows int8 training/inference, etc.'
    )
    parser.add_argument(
        "--siglip",
        default=False,
        action="store_true",
        help='Use SigLip (sigmoid) loss.'
    )
    parser.add_argument(
        "--dhnnce-loss",
        default=False,
        action="store_true",
        help="Use our DHN-NCE loss for training.",
    )
    parser.add_argument(
        "--temperature-dhnnce",
        type=float,
        default=0.6,
        help="Temperature for DHN-NCE loss.",
    )
    parser.add_argument(
        "--alpha-dhnnce",
        type=float,
        default=0.0,
        help="Alpha for DHN-NCE loss.",
    )
    parser.add_argument(
        "--beta1-dhnnce",
        type=float,
        default=0.15,
        help="Beta1 for DHN-NCE loss.",
    )
    parser.add_argument(
        "--beta2-dhnnce",
        type=float,
        default=0.15,
        help="Beta2 for DHN-NCE loss.",
    )

    args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    args.atomic_attrs = [x.strip() for x in args.atomic_attrs.split(",") if x.strip()]
    if args.num_atomic_attrs is None:
        args.num_atomic_attrs = len(args.atomic_attrs)

    preset_cfg = get_atomic_loss_preset(args.atomic_loss_preset)
    args.atomic_loss_preset_name = preset_cfg["name"]

    if not _flag_was_passed(cli_args, "--dpo-loss-type") and "dpo_loss_type" in preset_cfg:
        args.dpo_loss_type = preset_cfg["dpo_loss_type"]
    if args.lambda_global is None:
        args.lambda_global = preset_cfg["lambda_global"]
    if (
        not _flag_was_passed(cli_args, "--global-fourway-weights")
        and "global_fourway_weights" in preset_cfg
    ):
        args.global_fourway_weights = list(preset_cfg["global_fourway_weights"])
    if args.lambda_atomic_fourway is None:
        args.lambda_atomic_fourway = preset_cfg.get("lambda_atomic_fourway", 0.0)
    if (
        not _flag_was_passed(cli_args, "--atomic-fourway-weights")
        and "atomic_fourway_weights" in preset_cfg
    ):
        args.atomic_fourway_weights = list(preset_cfg["atomic_fourway_weights"])
    if args.lambda_atomic_text is None:
        args.lambda_atomic_text = preset_cfg["lambda_atomic_text"]
    if args.lambda_atomic_visual is None:
        args.lambda_atomic_visual = preset_cfg["lambda_atomic_visual"]

    if not _flag_was_passed(cli_args, "--atomic-train-mode"):
        args.atomic_train_mode = preset_cfg["atomic_train_mode"]
    if not _flag_was_passed(cli_args, "--atomic-text-weight-mode"):
        args.atomic_text_weight_mode = preset_cfg["atomic_text_weight_mode"]
    if not _flag_was_passed(cli_args, "--atomic-visual-use-neg-image"):
        args.atomic_visual_use_neg_image = preset_cfg["atomic_visual_use_neg_image"]
    if not _flag_was_passed(cli_args, "--atomic-visual-neg-source") and "atomic_visual_neg_source" in preset_cfg:
        args.atomic_visual_neg_source = preset_cfg["atomic_visual_neg_source"]
    if not _flag_was_passed(cli_args, "--atomic-visual-match-pool") and "atomic_visual_match_pool" in preset_cfg:
        args.atomic_visual_match_pool = preset_cfg["atomic_visual_match_pool"]
    if (
        not _flag_was_passed(cli_args, "--atomic-visual-max-other-mismatches")
        and "atomic_visual_max_other_mismatches" in preset_cfg
    ):
        args.atomic_visual_max_other_mismatches = preset_cfg["atomic_visual_max_other_mismatches"]
    if not _flag_was_passed(cli_args, "--scale-global-in-prop-sample"):
        args.scale_global_in_prop_sample = preset_cfg["scale_global_in_prop_sample"]
    if not _flag_was_passed(cli_args, "--atomic-dropout-k"):
        args.atomic_dropout_k = preset_cfg["atomic_dropout_k"]
    if not _flag_was_passed(cli_args, "--atomic-dropout-p"):
        args.atomic_dropout_p = preset_cfg["atomic_dropout_p"]
    if not _flag_was_passed(cli_args, "--hard-mining") and "hard_mining" in preset_cfg:
        args.hard_mining = preset_cfg["hard_mining"]
    if not _flag_was_passed(cli_args, "--hard-temperature") and "hard_temperature" in preset_cfg:
        args.hard_temperature = preset_cfg["hard_temperature"]
    if not _flag_was_passed(cli_args, "--hard-topk") and "hard_topk" in preset_cfg:
        args.hard_topk = preset_cfg["hard_topk"]
    if not _flag_was_passed(cli_args, "--semantic-weight-scale") and "semantic_weight_scale" in preset_cfg:
        args.semantic_weight_scale = preset_cfg["semantic_weight_scale"]
    if not _flag_was_passed(cli_args, "--semantic-weight-min") and "semantic_weight_min" in preset_cfg:
        args.semantic_weight_min = preset_cfg["semantic_weight_min"]
    if not _flag_was_passed(cli_args, "--semantic-weight-max") and "semantic_weight_max" in preset_cfg:
        args.semantic_weight_max = preset_cfg["semantic_weight_max"]
    if not _flag_was_passed(cli_args, "--attr-loss-weights") and preset_cfg.get("attr_loss_weights") is not None:
        args.attr_loss_weights = list(preset_cfg["attr_loss_weights"])

    default_attr_prior = {
        "location": 1.0,
        "size": 1.0,
        "shape": 1.2,
        "orientation": 1.2,
        "boundary": 1.3,
        "disease_type": 1.3,
    }
    if args.atomic_text_weight_mode == "attr_prior" and args.attr_loss_weights is None:
        missing = [attr for attr in args.atomic_attrs if attr not in default_attr_prior]
        if missing:
            parser.error(
                "No default attr prior exists for: " + ",".join(missing) + ". "
                "Pass --attr-loss-weights explicitly."
            )
        args.attr_loss_weights = [default_attr_prior[attr] for attr in args.atomic_attrs]

    if args.attr_loss_weights is not None:
        if isinstance(args.attr_loss_weights, str):
            args.attr_loss_weights = [float(x.strip()) for x in args.attr_loss_weights.split(",") if x.strip()]
        else:
            args.attr_loss_weights = [float(x) for x in args.attr_loss_weights]
        if len(args.attr_loss_weights) != len(args.atomic_attrs):
            parser.error("--attr-loss-weights must match the number of --atomic-attrs")

    if args.atomic_dropout_k > 0 and args.atomic_dropout_p > 0:
        parser.error("--atomic-dropout-k and --atomic-dropout-p cannot both be enabled")
    if args.atomic_dropout_k < 0:
        parser.error("--atomic-dropout-k must be >= 0")
    if not 0.0 <= args.atomic_dropout_p <= 1.0:
        parser.error("--atomic-dropout-p must be in [0, 1]")
    if args.atomic_visual_max_other_mismatches < 0:
        parser.error("--atomic-visual-max-other-mismatches must be >= 0")
    if args.num_atomic_attrs <= 0:
        parser.error("--num-atomic-attrs must be > 0")
    if args.hard_mining == "topk" and args.hard_topk <= 0:
        parser.error("--hard-topk must be > 0 when --hard-mining=topk")
    if args.atomic_text_weight_mode == "semantic_ref_cache" and not args.semantic_weight_cache_path:
        parser.error("--semantic-weight-cache-path is required when --atomic-text-weight-mode=semantic_ref_cache")
    if args.semantic_weight_min > args.semantic_weight_max:
        parser.error("--semantic-weight-min must be <= --semantic-weight-max")
    if args.atomic_visual_neg_source == "atomic_matched":
        if args.dpo_mode != "atomic":
            parser.error("--atomic-visual-neg-source=atomic_matched requires --dpo-mode=atomic")
        if args.atomic_train_mode != "row_all":
            parser.error("--atomic-visual-neg-source=atomic_matched currently supports only --atomic-train-mode=row_all")
        if not args.atomic_visual_use_neg_image:
            parser.error("--atomic-visual-neg-source=atomic_matched requires --atomic-visual-use-neg-image=true")
        if not args.atomic_visual_attr_json_path:
            parser.error("--atomic-visual-attr-json-path is required when --atomic-visual-neg-source=atomic_matched")

    return args
