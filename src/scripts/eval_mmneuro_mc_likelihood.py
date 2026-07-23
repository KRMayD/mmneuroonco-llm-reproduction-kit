#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.pl_pathflip_finetune import pl_pathflip_finetune


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    return data


def get_instruction(sample):
    return str(sample["conversations"][0]["value"])


def get_target(sample):
    return str(sample["conversations"][1]["value"]).strip().upper()


def get_images(sample):
    images = sample.get("image") or sample.get("images") or []
    return [images] if isinstance(images, str) else [str(item) for item in images]


@torch.no_grad()
def score_candidates(core_model, image_paths, instruction, candidates):
    visual_tokens, visual_mask = core_model._get_visual_tokens(
        {"raw_image_paths": [list(image_paths)]}
    )
    visual_tokens = visual_tokens.repeat(len(candidates), 1, 1)
    visual_mask = visual_mask.repeat(len(candidates), 1)

    prompts = [core_model._format_instruction_prompt(instruction) for _ in candidates]
    prompt_embeds, prompt_mask = core_model._build_prompt_inputs(
        prompts, visual_tokens, visual_mask
    )
    candidate_tokens = core_model.llm_tokenizer(
        list(candidates), padding=True, truncation=True, return_tensors="pt"
    ).to(core_model.device)
    candidate_embeds = core_model.llm_model.get_input_embeddings()(
        candidate_tokens.input_ids
    )
    inputs_embeds = torch.cat([prompt_embeds, candidate_embeds], dim=1)
    attention_mask = torch.cat([prompt_mask, candidate_tokens.attention_mask], dim=1)

    ignored = torch.full(
        (len(candidates), prompt_embeds.size(1)),
        -100,
        device=core_model.device,
        dtype=torch.long,
    )
    candidate_labels = candidate_tokens.input_ids.masked_fill(
        candidate_tokens.attention_mask.eq(0), -100
    )
    labels = torch.cat([ignored, candidate_labels], dim=1)

    outputs = core_model.llm_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_log_probs = F.log_softmax(shift_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    scores_tensor = (token_log_probs * valid).sum(dim=1)
    scores = {
        candidate: float(score.detach().cpu())
        for candidate, score in zip(candidates, scores_tensor)
    }
    prediction = max(scores, key=scores.get)
    return prediction, scores


def build_summary(records):
    by_type = defaultdict(lambda: {"total": 0, "correct": 0, "predictions": Counter()})
    for record in records:
        slot = by_type[record["question_type"]]
        slot["total"] += 1
        slot["correct"] += int(record["correct"])
        slot["predictions"][record["prediction"]] += 1
    normalized = {}
    for question_type, slot in sorted(by_type.items()):
        normalized[question_type] = {
            "total": slot["total"],
            "correct": slot["correct"],
            "accuracy": slot["correct"] / slot["total"] if slot["total"] else 0.0,
            "prediction_distribution": dict(slot["predictions"]),
        }
    correct = sum(int(record["correct"]) for record in records)
    return {
        "evaluation": "MM-NeuroOnco Closed-VQA MC likelihood",
        "total": len(records),
        "correct": correct,
        "accuracy": correct / len(records) if records else 0.0,
        "by_question_type": normalized,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    model = pl_pathflip_finetune.load_from_checkpoint(
        str(args.checkpoint), map_location="cpu", strict=False
    )
    model.to(args.device)
    model.eval()
    core_model = model.pathflip_finetune

    data = read_json(args.test_json)
    data = [
        sample
        for sample in data
        if (sample.get("meta") or {}).get("source") == "MM-NeuroOnco_Benchmark_Closed"
    ]
    if args.limit:
        data = data[: args.limit]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for sample in tqdm(data, desc="Closed-VQA MC likelihood"):
            meta = sample.get("meta") or {}
            options = meta.get("options") or {}
            candidates = [key for key in ("A", "B", "C", "D", "E") if key in options]
            prediction, scores = score_candidates(
                core_model,
                get_images(sample),
                get_instruction(sample),
                candidates,
            )
            target = get_target(sample)
            record = {
                "id": meta.get("id"),
                "q_index": meta.get("q_index"),
                "question_type": meta.get("question_type", "unknown"),
                "question": meta.get("question", ""),
                "options": options,
                "target": target,
                "prediction": prediction,
                "correct": prediction == target,
                "scores": scores,
                "image": get_images(sample),
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    summary = build_summary(records)
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "test_json": str(args.test_json),
            "output_jsonl": str(args.output_jsonl),
        }
    )
    summary_path = Path(str(args.output_jsonl) + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
