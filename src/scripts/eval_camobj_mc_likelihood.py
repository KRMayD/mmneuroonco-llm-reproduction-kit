#!/usr/bin/env python3
"""Evaluate a PathFLIP checkpoint on MM-CamObj Easy/Hard VQA by MC likelihood."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.pl_pathflip_finetune import pl_pathflip_finetune


def read_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


@torch.no_grad()
def encode_image(core_model, image_path: Path):
    tokens, mask = core_model._get_visual_tokens({'raw_image_paths': [[str(image_path)]]})
    return tokens.detach().cpu(), mask.detach().cpu()


@torch.no_grad()
def score_candidates(core_model, cached_visual, instruction: str, candidates):
    visual_tokens, visual_mask = cached_visual
    visual_tokens = visual_tokens.to(core_model.device).repeat(len(candidates), 1, 1)
    visual_mask = visual_mask.to(core_model.device).repeat(len(candidates), 1)
    prompts = [core_model._format_instruction_prompt(instruction) for _ in candidates]
    prompt_embeds, prompt_mask = core_model._build_prompt_inputs(
        prompts, visual_tokens, visual_mask
    )
    candidate_tokens = core_model.llm_tokenizer(
        list(candidates), padding=True, truncation=True, return_tensors='pt'
    ).to(core_model.device)
    candidate_embeds = core_model.llm_model.get_input_embeddings()(candidate_tokens.input_ids)
    inputs_embeds = torch.cat([prompt_embeds, candidate_embeds], dim=1)
    attention_mask = torch.cat([prompt_mask, candidate_tokens.attention_mask], dim=1)
    ignored = torch.full(
        (len(candidates), prompt_embeds.size(1)), -100, device=core_model.device, dtype=torch.long
    )
    candidate_labels = candidate_tokens.input_ids.masked_fill(
        candidate_tokens.attention_mask.eq(0), -100
    )
    labels = torch.cat([ignored, candidate_labels], dim=1)
    logits = core_model.llm_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    ).logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_log_probs = F.log_softmax(logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    values = (token_log_probs * valid).sum(dim=1)
    scores = {candidate: float(value.detach().cpu()) for candidate, value in zip(candidates, values)}
    return max(scores, key=scores.get), scores


def macro_metrics(records):
    classes = ('A', 'B', 'C', 'D')
    total = len(records)
    correct = sum(record['correct'] for record in records)
    metrics = []
    for label in classes:
        tp = sum(record['target'] == label and record['prediction'] == label for record in records)
        fp = sum(record['target'] != label and record['prediction'] == label for record in records)
        fn = sum(record['target'] == label and record['prediction'] != label for record in records)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics.append((precision, recall, f1))
    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total else 0.0,
        'macro_precision': sum(item[0] for item in metrics) / len(metrics),
        'macro_recall': sum(item[1] for item in metrics) / len(metrics),
        'macro_f1': sum(item[2] for item in metrics) / len(metrics),
        'prediction_distribution': dict(Counter(record['prediction'] for record in records)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-jsonl', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    model = pl_pathflip_finetune.load_from_checkpoint(str(args.checkpoint), map_location='cpu', strict=False)
    model.to(args.device)
    model.eval()
    core_model = model.pathflip_finetune

    questions = read_jsonl(args.questions)
    if args.limit:
        questions = questions[:args.limit]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    visual_cache = {}
    records = []
    with args.output_jsonl.open('w', encoding='utf-8') as handle:
        for item in tqdm(questions, desc=f'MC likelihood: {args.questions.stem}'):
            image_path = (args.dataset_root / item['image'].removeprefix('data/')).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f'Missing question image: {image_path}')
            image_key = str(image_path)
            if image_key not in visual_cache:
                visual_cache[image_key] = encode_image(core_model, image_path)
            prediction, scores = score_candidates(
                core_model, visual_cache[image_key], item['text'], ('A', 'B', 'C', 'D')
            )
            record = {
                'question_id': item['question_id'],
                'image': image_key,
                'category': item.get('category'),
                'question': item['text'],
                'target': item['answer'],
                'prediction': prediction,
                'correct': prediction == item['answer'],
                'scores': scores,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    summary = macro_metrics(records)
    summary.update({
        'evaluation': 'MM-CamObj MC likelihood',
        'checkpoint': str(args.checkpoint),
        'questions': str(args.questions),
        'dataset_root': str(args.dataset_root),
        'output_jsonl': str(args.output_jsonl),
        'unique_encoded_images': len(visual_cache),
    })
    summary_path = Path(str(args.output_jsonl) + '.summary.json')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
