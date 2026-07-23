"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import sys
import json
import hashlib
import shutil
import tempfile
from typing import Sequence

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .blip2 import Blip2Base

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_default_dtype(torch.bfloat16)


DEFAULT_BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DEFAULT_OPEN_CLIP_SRC = ""
DEFAULT_BIOMEDCLIP_IMAGE_SIZE = 224
DEFAULT_BIOMEDCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
DEFAULT_BIOMEDCLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_tokenizer(pretrained_model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="right")
    tokenizer.add_special_tokens(
        {
            "pad_token": "[pad]",
            "bos_token": "[bos]",
            "unk_token": "[unk]",
            "additional_special_tokens": ["<image>"],
        }
    )
    return tokenizer


def get_prompt(prompt_type="generate_caption"):
    if prompt_type == "generate_caption":
        prompt = (
            "Instruction: Describe the input pathology whole slide image.\n"
            "Input pathology whole slide image: <image>.\n"
            "Response: "
        )
        return prompt
    raise NotImplementedError()


class pathflip_finetune(Blip2Base):
    def __init__(
        self,
        bert_name="/path/bert-base-uncased",
        text_max_length=512,
        path_enc="Linear",
        num_query_token=32,
        num_hidden_layers=12,
        cross_attention_freq=2,
        path_input_dim=512,
        embed_dim=256,
        llm_model="/path/Qwen3-0.6B",
        llm_tuning="lora",
        caption_prompt=None,
        args=None,
    ):
        super().__init__()

        self.args = args
        self.path_input_dim = path_input_dim
        self.text_max_length = text_max_length
        self.generate_caption_prompt = caption_prompt or get_prompt(prompt_type="generate_caption")

        # Keep the original modules instantiated for compatibility, but the
        # active visual path no longer uses them.
        self.tokenizer = self.init_tokenizer(bert_name)
        self.path_encoder = self.init_path_encoder(
            input_dim=path_input_dim,
            emb_dim=embed_dim,
            model_name=path_enc,
        )
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token=num_query_token,
            vision_width=embed_dim,
            num_hidden_layers=num_hidden_layers,
            cross_attention_freq=cross_attention_freq,
            bert_name=bert_name,
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.path_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        trans_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.path_trans = nn.TransformerEncoder(trans_layer, num_layers=2)

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype="auto",
            device_map=None,
        )
        config = AutoConfig.from_pretrained(llm_model)
        llm_hidden_size = config.hidden_size

        self.llm_tokenizer = get_tokenizer(llm_model)
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        self.llm_tokenizer.image_token_id = self.llm_tokenizer(
            "<image>", add_special_tokens=False
        ).input_ids[0]

        lora_r = 16
        lora_alpha = 32
        lora_dropout = 0.05
        if llm_tuning == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=[
                    "k_proj",
                    "v_proj",
                    "q_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            self.peft_config = peft_config
            self.llm_model = get_peft_model(self.llm_model, peft_config)
            self.llm_model.print_trainable_parameters()
        elif llm_tuning == "full":
            pass
        elif llm_tuning == "freeze":
            for _, param in self.llm_model.named_parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError()

        self.eos_token_id = self.llm_tokenizer.eos_token_id
        self.pad_token_id = self.llm_tokenizer.pad_token_id
        self.path_proj_llm = nn.Linear(path_input_dim, llm_hidden_size)

        self.biomedclip_model_name = getattr(args, "biomedclip_model_name", DEFAULT_BIOMEDCLIP_MODEL)
        self.biomedclip_pretrained = getattr(args, "biomedclip_pretrained", "")
        self.biomedclip_open_clip_src = getattr(args, "biomedclip_open_clip_src", DEFAULT_OPEN_CLIP_SRC)
        self.normalize_visual_features = bool(
            getattr(args, "normalize_visual_features", False)
        )
        self.biomedclip_model = None
        self.biomedclip_preprocess = None
        self.biomedclip_runtime_dir = None

    def _resolve_biomedclip_pretrained_path(self, pretrained):
        if os.path.isdir(pretrained):
            for candidate_name in ("pytorch_model.bin", "open_clip_pytorch_model.bin"):
                candidate_path = os.path.join(pretrained, candidate_name)
                if os.path.exists(candidate_path):
                    return candidate_path
            raise FileNotFoundError(
                f"Could not find a BiomedCLIP checkpoint file under directory: {pretrained}"
            )
        return pretrained

    def _load_biomedclip_state_dict(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported BiomedCLIP checkpoint format at {checkpoint_path}")

        normalized_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                key = key[len("model.") :]
            if key.startswith("module."):
                key = key[len("module.") :]
            normalized_state_dict[key] = value
        return normalized_state_dict

    def _is_hf_style_biomedclip_state_dict(self, state_dict):
        sample_keys = list(state_dict.keys())[:64]
        hf_prefixes = (
            "vision_model.",
            "text_model.",
            "visual_projection.",
            "text_projection.",
        )
        return any(key.startswith(hf_prefixes) for key in sample_keys)

    def _find_biomedclip_support_dir(self, pretrained_path):
        search_dir = pretrained_path if os.path.isdir(pretrained_path) else os.path.dirname(pretrained_path)
        required_files = ("config.json", "configuration_biomed_clip.py", "modeling_biomed_clip.py")

        while search_dir and search_dir != os.path.dirname(search_dir):
            if all(os.path.exists(os.path.join(search_dir, filename)) for filename in required_files):
                return search_dir
            search_dir = os.path.dirname(search_dir)

        raise FileNotFoundError(
            "Could not locate BiomedCLIP custom modeling files "
            f"starting from {pretrained_path}"
        )

    def _find_biomedclip_preprocess_config(self, pretrained_path, support_dir):
        search_dirs = []
        pretrained_dir = pretrained_path if os.path.isdir(pretrained_path) else os.path.dirname(pretrained_path)
        search_dirs.append(pretrained_dir)
        if support_dir not in search_dirs:
            search_dirs.append(support_dir)

        for search_dir in search_dirs:
            config_path = os.path.join(search_dir, "open_clip_config.json")
            if os.path.exists(config_path):
                return config_path
        return None

    def _build_biomedclip_preprocess(self, pretrained_path, support_dir):
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        image_size = DEFAULT_BIOMEDCLIP_IMAGE_SIZE
        mean = DEFAULT_BIOMEDCLIP_MEAN
        std = DEFAULT_BIOMEDCLIP_STD

        preprocess_config_path = self._find_biomedclip_preprocess_config(pretrained_path, support_dir)
        if preprocess_config_path is not None:
            with open(preprocess_config_path, "r", encoding="utf-8") as f:
                preprocess_config = json.load(f)
            preprocess_cfg = preprocess_config.get("preprocess_cfg", {})
            model_cfg = preprocess_config.get("model_cfg", {})
            vision_cfg = model_cfg.get("vision_cfg", {})
            image_size = vision_cfg.get("image_size", image_size)
            mean = tuple(preprocess_cfg.get("mean", mean))
            std = tuple(preprocess_cfg.get("std", std))

        resize_size = tuple(image_size) if isinstance(image_size, (list, tuple)) else image_size
        crop_size = resize_size

        return transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def _prepare_biomedclip_runtime_dir(self, support_dir, checkpoint_path, state_dict):
        cache_root = os.path.join(tempfile.gettempdir(), "pathflip_biomedclip_runtime")
        os.makedirs(cache_root, exist_ok=True)
        stat = os.stat(checkpoint_path)
        cache_source = f"{os.path.abspath(checkpoint_path)}:{stat.st_mtime_ns}:{stat.st_size}"
        cache_key = hashlib.sha1(cache_source.encode("utf-8")).hexdigest()[:16]
        runtime_dir = os.path.join(cache_root, cache_key)
        os.makedirs(runtime_dir, exist_ok=True)

        for filename in ("config.json", "configuration_biomed_clip.py", "modeling_biomed_clip.py"):
            src_path = os.path.join(support_dir, filename)
            dst_path = os.path.join(runtime_dir, filename)
            if os.path.exists(dst_path):
                continue
            try:
                os.symlink(src_path, dst_path)
            except OSError:
                shutil.copy2(src_path, dst_path)

        weights_path = os.path.join(runtime_dir, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            torch.save(state_dict, weights_path)

        return runtime_dir

    def _load_hf_style_biomedclip(self, checkpoint_path, state_dict):
        support_dir = self._find_biomedclip_support_dir(checkpoint_path)
        runtime_dir = self._prepare_biomedclip_runtime_dir(support_dir, checkpoint_path, state_dict)
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            runtime_dir,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        model = model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.biomedclip_model = model
        self.biomedclip_preprocess = self._build_biomedclip_preprocess(checkpoint_path, support_dir)
        self.biomedclip_runtime_dir = runtime_dir

    def _load_biomedclip(self):
        if self.biomedclip_model is not None:
            self.biomedclip_model = self.biomedclip_model.to(self.device)
            return

        pretrained = self.biomedclip_pretrained or None
        if pretrained and os.path.exists(pretrained):
            checkpoint_path = self._resolve_biomedclip_pretrained_path(pretrained)
            state_dict = self._load_biomedclip_state_dict(checkpoint_path)
            if self._is_hf_style_biomedclip_state_dict(state_dict):
                self._load_hf_style_biomedclip(checkpoint_path, state_dict)
                return
            pretrained = checkpoint_path

        if self.biomedclip_open_clip_src and self.biomedclip_open_clip_src not in sys.path:
            sys.path.insert(0, self.biomedclip_open_clip_src)

        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "Failed to import open_clip for BiomedCLIP loading. "
                "Install open_clip_torch or set BIOMEDCLIP_OPEN_CLIP_SRC to a vendored open_clip/src directory. "
                f"Checked source path: {self.biomedclip_open_clip_src}"
            ) from exc

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.biomedclip_model_name,
            pretrained=pretrained,
            device=self.device,
            precision="fp32",
        )
        model = model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.biomedclip_model = model
        self.biomedclip_preprocess = preprocess

    def _prepare_precomputed_visual_tokens(self, path_features, path_mask=None):
        if path_features.dim() == 2:
            visual_tokens = path_features.unsqueeze(1)
            visual_mask = torch.ones(
                (path_features.size(0), 1),
                dtype=torch.long,
                device=self.device,
            )
        elif path_features.dim() == 3:
            visual_tokens = path_features
            if path_mask is None:
                visual_mask = torch.ones(
                    visual_tokens.shape[:2],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                visual_mask = path_mask.to(device=self.device, dtype=torch.long)
        else:
            raise ValueError(
                "Expected batch['path'] to have shape [B, D] or [B, T, D], "
                f"but got {tuple(path_features.shape)}."
            )

        if visual_tokens.size(-1) != self.path_input_dim:
            raise ValueError(
                f"Visual feature dim mismatch: expected {self.path_input_dim}, "
                f"got {visual_tokens.size(-1)}."
            )

        return (
            visual_tokens.to(device=self.device, dtype=self.path_proj_llm.weight.dtype),
            visual_mask,
        )

    def _encode_raw_images(self, raw_image_paths: Sequence[Sequence[str]]):
        self._load_biomedclip()

        batch_features = []
        with torch.no_grad():
            for sample_paths in raw_image_paths:
                if isinstance(sample_paths, str):
                    sample_paths = [sample_paths]
                if not sample_paths:
                    raise ValueError("Received an empty raw_image_paths entry.")

                sample_images = []
                for image_path in sample_paths:
                    with Image.open(image_path) as image:
                        sample_images.append(self.biomedclip_preprocess(image.convert("RGB")))

                pixel_values = torch.stack(sample_images, dim=0).to(self.device)
                if hasattr(self.biomedclip_model, "encode_image"):
                    image_features = self.biomedclip_model.encode_image(pixel_values)
                else:
                    image_features = self.biomedclip_model.get_image_features(pixel_values=pixel_values)
                if self.normalize_visual_features:
                    image_features = F.normalize(image_features.float(), p=2, dim=-1).to(
                        image_features.dtype
                    )
                sample_feature = image_features.mean(dim=0)
                if self.normalize_visual_features:
                    sample_feature = F.normalize(sample_feature.float(), p=2, dim=-1).to(
                        image_features.dtype
                    )
                batch_features.append(sample_feature)

        visual_features = torch.stack(batch_features, dim=0)
        return self._prepare_precomputed_visual_tokens(visual_features)

    def _get_visual_tokens(self, batch):
        path = batch.get("path")
        if path is not None:
            return self._prepare_precomputed_visual_tokens(path, batch.get("path_mask"))

        raw_image_paths = batch.get("raw_image_paths")
        if raw_image_paths is not None:
            return self._encode_raw_images(raw_image_paths)

        raise KeyError("Expected batch to contain either 'path' or 'raw_image_paths'.")

    def _build_prompt_inputs(self, prompt, visual_tokens, visual_mask):
        batch_size = visual_tokens.size(0)
        device = self.device

        if isinstance(prompt, (list, tuple)):
            if len(prompt) != batch_size:
                raise ValueError(f"Expected {batch_size} prompts, got {len(prompt)}.")
            prompt_tokens = self.llm_tokenizer(
                list(prompt),
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            image_positions = []
            for row in range(batch_size):
                positions = (
                    prompt_tokens.input_ids[row] == self.llm_tokenizer.image_token_id
                ).nonzero(as_tuple=True)[0]
                if positions.numel() != 1:
                    raise ValueError(
                        f"Expected exactly one <image> placeholder, got {positions.numel()}."
                    )
                image_positions.append(int(positions.item()))
            if len(set(image_positions)) != 1:
                raise ValueError("Batched prompts must place <image> at the same token position.")
            image_pos = image_positions[0]
            prompt_input_ids = prompt_tokens.input_ids
            prompt_attention_mask = prompt_tokens.attention_mask
        else:
            prompt_tokens = self.llm_tokenizer(
                prompt,
                padding=False,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            image_positions = (
                prompt_tokens.input_ids == self.llm_tokenizer.image_token_id
            ).nonzero(as_tuple=True)[1]
            if image_positions.numel() != 1:
                raise ValueError(
                    f"Expected exactly one <image> placeholder, got {image_positions.numel()}."
                )
            image_pos = int(image_positions.item())
            prompt_input_ids = prompt_tokens.input_ids.repeat(batch_size, 1)
            prompt_attention_mask = prompt_tokens.attention_mask.repeat(batch_size, 1)
        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_input_ids)
        visual_embeds = self.path_proj_llm(visual_tokens)
        visual_embeds = visual_embeds.to(dtype=prompt_embeds.dtype)

        inputs_embeds = torch.cat(
            [
                prompt_embeds[:, :image_pos, :],
                visual_embeds,
                prompt_embeds[:, image_pos + 1 :, :],
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                prompt_attention_mask[:, :image_pos],
                visual_mask,
                prompt_attention_mask[:, image_pos + 1 :],
            ],
            dim=1,
        )
        return inputs_embeds, attention_mask

    def _format_instruction_prompt(self, instruction):
        instruction = (instruction or "").replace("<image>", "").strip()
        return f"Input medical image: <image>.\nInstruction: {instruction}\nResponse: "

    def forward(self, batch, return_attn=False):
        del return_attn

        text = batch["text"]
        device = self.device
        visual_tokens, visual_mask = self._get_visual_tokens(batch)
        batch_size = visual_tokens.size(0)

        prompt = self.generate_caption_prompt
        if "instruction" in batch:
            instructions = batch["instruction"]
            if isinstance(instructions, str):
                prompt = self._format_instruction_prompt(instructions)
            else:
                prompt = [self._format_instruction_prompt(item) for item in instructions]

        inputs_embeds_1, attention_mask_1 = self._build_prompt_inputs(
            prompt, visual_tokens, visual_mask
        )

        text_tokens = self.llm_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        ).to(device)

        inputs_embeds_2 = self.llm_model.get_input_embeddings()(text_tokens.input_ids)
        attention_mask_2 = text_tokens.attention_mask

        inputs_embeds = torch.cat([inputs_embeds_1, inputs_embeds_2], dim=1)
        attention_mask = torch.cat([attention_mask_1, attention_mask_2], dim=1)

        targets_1 = torch.full((batch_size, inputs_embeds_1.size(1)), -100, device=device, dtype=torch.long)
        targets_2 = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
        )
        targets = torch.cat([targets_1, targets_2], dim=1)

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=targets,
            use_cache=True,
            return_dict=True,
        )
        return outputs.loss

    def forward_image(self, batch):
        visual_tokens, visual_mask = self._get_visual_tokens(batch)
        weighted_sum = (visual_tokens * visual_mask.unsqueeze(-1)).sum(dim=1)
        denom = visual_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return weighted_sum / denom

    def forward_text(self, batch):
        text = batch["text"]
        device = self.device

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        ).to(device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return self.text_proj(text_output.last_hidden_state[:, 0, :])

    @torch.no_grad()
    def generate(
        self,
        batch,
        do_sample=True,
        use_nucleus_sampling=False,
        num_beams=3,
        max_new_tokens=512,
        min_new_tokens=128,
        length_penalty=1.0,
        repetition_penalty=1.0,
        num_captions=3,
    ):
        del use_nucleus_sampling

        visual_tokens, visual_mask = self._get_visual_tokens(batch)
        inputs_embeds, attention_mask = self._build_prompt_inputs(
            self.generate_caption_prompt, visual_tokens, visual_mask
        )

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        return self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    @torch.no_grad()
    def generate_with_instruction(
        self,
        batch,
        do_sample=True,
        use_nucleus_sampling=False,
        num_beams=3,
        max_new_tokens=128,
        length_penalty=1.0,
        repetition_penalty=1.0,
        num_captions=3,
    ):
        del use_nucleus_sampling

        instruction = batch["instruction"]
        if isinstance(instruction, str):
            prompt = self._format_instruction_prompt(instruction)
        else:
            prompt = [self._format_instruction_prompt(item) for item in instruction]
        visual_tokens, visual_mask = self._get_visual_tokens(batch)
        inputs_embeds, attention_mask = self._build_prompt_inputs(prompt, visual_tokens, visual_mask)

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )
        return self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    from utils.process_args import get_args

    import logging

    logging.getLogger("transformers").setLevel(logging.ERROR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    text = [
        "The pathological findings reveal that the patient has an invasive high-grade urothelial carcinoma."
    ]
    batch = {
        "path": torch.randn(1, args.path_input_dim).to(device),
        "text": text,
    }

    model = pathflip_finetune(args=args).to(device)
    loss = model(batch)
    print(f"loss: {loss}")

    generated_text = model.generate(batch, num_captions=1, max_new_tokens=16, min_new_tokens=1)
    print(f"generated_text: {generated_text}")
