#!/usr/bin/env python3
"""Generate and score MM-NeuroOnco Open-VQA responses.

The evaluator intentionally keeps decoding deterministic and reports three
complementary measures: BERTScore F1, reference-keyword recall, and numeric
value accuracy. It is designed for comparing vision encoders under an
otherwise identical downstream MLLM setup.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

# The environment contains an optional scikit-learn wheel incompatible with
# its NumPy version. Transformers only needs sklearn for optional assisted
# generation helpers, so expose it as unavailable before Transformers loads.
_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def _find_spec_without_sklearn(name: str, package: str | None = None):
    if name == "sklearn" or name.startswith("sklearn."):
        return None
    return _ORIGINAL_FIND_SPEC(name, package)


importlib.util.find_spec = _find_spec_without_sklearn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.pl_pathflip_finetune import pl_pathflip_finetune  # noqa: E402


NUMBER_RE = re.compile(
    r"(?:(?P<axis>[xy])\s*=\s*)?(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<percent>%?)",
    flags=re.IGNORECASE,
)
TOKEN_RE = re.compile(r"\s+")


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def get_images(sample: dict[str, Any]) -> list[str]:
    images = sample.get("image") or sample.get("images") or []
    return [str(images)] if isinstance(images, str) else [str(item) for item in images]


def get_instruction(sample: dict[str, Any]) -> str:
    return str(sample["conversations"][0]["value"])


def get_reference(sample: dict[str, Any]) -> str:
    meta = sample.get("meta") or {}
    return str(meta.get("reference_answer") or sample["conversations"][1]["value"])


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"[^a-z0-9.%+\-\s]", " ", value)
    return TOKEN_RE.sub(" ", value).strip()


def phrase_present(phrase: str, text: str) -> bool:
    normalized_phrase = normalize_text(phrase)
    normalized_text = normalize_text(text)
    if not normalized_phrase:
        return False
    return re.search(
        rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])",
        normalized_text,
    ) is not None


def extract_numbers(text: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for match in NUMBER_RE.finditer(normalize_text(text)):
        axis = (match.group("axis") or "").lower() or None
        raw_value = match.group("value")
        values.append(
            {
                "value": float(raw_value),
                "axis": axis,
                "percent": bool(match.group("percent")),
                "decimal": "." in raw_value,
            }
        )
    return values


def numeric_tolerance(number: dict[str, Any], args: argparse.Namespace) -> float:
    if number["percent"]:
        return args.numeric_percent_tolerance
    if number["axis"] is not None:
        return args.numeric_coordinate_tolerance
    if number["decimal"]:
        return args.numeric_decimal_tolerance
    return args.numeric_integer_tolerance


def numeric_matches(
    reference_numbers: list[dict[str, Any]],
    prediction_numbers: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[bool]:
    matches: list[bool] = []
    for reference in reference_numbers:
        compatible = [
            candidate
            for candidate in prediction_numbers
            if candidate["percent"] == reference["percent"]
            and (reference["axis"] is None or candidate["axis"] == reference["axis"])
        ]
        tolerance = numeric_tolerance(reference, args)
        matches.append(
            any(abs(candidate["value"] - reference["value"]) <= tolerance for candidate in compatible)
        )
    return matches


def batched(values: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def generate_records(
    model: pl_pathflip_finetune,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    core_model = model.pathflip_finetune
    records: list[dict[str, Any]] = []
    for batch_samples in tqdm(list(batched(samples, args.generation_batch_size)), desc="Open-VQA generation"):
        batch = {
            "raw_image_paths": [get_images(sample) for sample in batch_samples],
            "instruction": [get_instruction(sample) for sample in batch_samples],
        }
        predictions = core_model.generate_with_instruction(
            batch,
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            num_captions=1,
        )
        if len(predictions) != len(batch_samples):
            raise RuntimeError(
                f"Generation returned {len(predictions)} outputs for {len(batch_samples)} inputs."
            )
        for sample, prediction in zip(batch_samples, predictions):
            meta = sample.get("meta") or {}
            reference = get_reference(sample)
            keywords = [str(item) for item in meta.get("eval_keywords") or []]
            reference_numbers = extract_numbers(reference)
            prediction_numbers = extract_numbers(prediction)
            number_matches = numeric_matches(reference_numbers, prediction_numbers, args)
            matched_keywords = [keyword for keyword in keywords if phrase_present(keyword, prediction)]
            records.append(
                {
                    "id": meta.get("id"),
                    "q_index": meta.get("q_index"),
                    "question_id": meta.get("question_id"),
                    "question_type": meta.get("question_type", "unknown"),
                    "question": meta.get("question") or get_instruction(sample),
                    "reference": reference,
                    "prediction": prediction,
                    "eval_keywords": keywords,
                    "matched_keywords": matched_keywords,
                    "keyword_recall": len(matched_keywords) / len(keywords) if keywords else None,
                    "reference_numbers": reference_numbers,
                    "prediction_numbers": prediction_numbers,
                    "numeric_matches": number_matches,
                    "image": get_images(sample),
                }
            )
    return records


def add_bertscore(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    try:
        from bert_score import score as bert_score
    except Exception as exc:  # pragma: no cover - dependency/runtime diagnostic
        raise RuntimeError(
            "BERTScore could not be imported. Use the local safetensors model and "
            "the built-in sklearn compatibility shim, or pass --skip-bertscore."
        ) from exc

    predictions = [record["prediction"] for record in records]
    references = [record["reference"] for record in records]
    precision, recall, f1 = bert_score(
        predictions,
        references,
        model_type=str(args.bertscore_model),
        num_layers=args.bertscore_num_layers,
        lang="en",
        device=args.device,
        batch_size=args.bertscore_batch_size,
        rescale_with_baseline=False,
        verbose=True,
    )
    for record, value_p, value_r, value_f1 in zip(records, precision, recall, f1):
        record["bertscore_precision"] = float(value_p.cpu())
        record["bertscore_recall"] = float(value_r.cpu())
        record["bertscore_f1"] = float(value_f1.cpu())


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    keyword_values = [record["keyword_recall"] for record in records if record["keyword_recall"] is not None]
    numeric_records = [record for record in records if record["reference_numbers"]]
    numeric_total = sum(len(record["reference_numbers"]) for record in numeric_records)
    numeric_correct = sum(sum(record["numeric_matches"]) for record in numeric_records)
    return {
        "total": len(records),
        "bertscore_precision": mean([record["bertscore_precision"] for record in records if "bertscore_precision" in record]),
        "bertscore_recall": mean([record["bertscore_recall"] for record in records if "bertscore_recall" in record]),
        "bertscore_f1": mean([record["bertscore_f1"] for record in records if "bertscore_f1" in record]),
        "keyword_recall_macro": mean(keyword_values),
        "keyword_samples": len(keyword_values),
        "numeric_samples": len(numeric_records),
        "numeric_value_accuracy": numeric_correct / numeric_total if numeric_total else None,
        "numeric_sample_accuracy": (
            sum(all(record["numeric_matches"]) for record in numeric_records) / len(numeric_records)
            if numeric_records
            else None
        ),
        "numeric_values_total": numeric_total,
    }


def build_summary(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[record["question_type"]].append(record)
    return {
        "evaluation": "MM-NeuroOnco Open-VQA deterministic generation",
        "total": len(records),
        "primary_metrics": ["bertscore_f1", "keyword_recall_macro", "numeric_value_accuracy"],
        "overall": summarize_group(records),
        "by_question_type": {question_type: summarize_group(group) for question_type, group in sorted(by_type.items())},
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "generation_batch_size": args.generation_batch_size,
        },
        "numeric_matching": {
            "percent_absolute_tolerance": args.numeric_percent_tolerance,
            "coordinate_absolute_tolerance": args.numeric_coordinate_tolerance,
            "decimal_absolute_tolerance": args.numeric_decimal_tolerance,
            "integer_absolute_tolerance": args.numeric_integer_tolerance,
        },
        "bertscore": {
            "model": str(args.bertscore_model),
            "num_layers": args.bertscore_num_layers,
            "rescale_with_baseline": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bertscore-model", type=Path, required=True)
    parser.add_argument("--bertscore-num-layers", type=int, default=12)
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--numeric-percent-tolerance", type=float, default=0.5)
    parser.add_argument("--numeric-coordinate-tolerance", type=float, default=2.0)
    parser.add_argument("--numeric-decimal-tolerance", type=float, default=0.02)
    parser.add_argument("--numeric-integer-tolerance", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-bertscore", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.bertscore_model.is_dir():
        raise FileNotFoundError(args.bertscore_model)

    samples = [
        sample
        for sample in read_json(args.test_json)
        if (sample.get("meta") or {}).get("source") == "MM-NeuroOnco_Benchmark_Open"
    ]
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("No MM-NeuroOnco Open-VQA samples found.")

    model = pl_pathflip_finetune.load_from_checkpoint(str(args.checkpoint), map_location="cpu", strict=False)
    model.to(args.device)
    model.eval()
    records = generate_records(model, samples, args)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.skip_bertscore:
        add_bertscore(records, args)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = build_summary(records, args)
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "test_json": str(args.test_json),
            "output_jsonl": str(args.output_jsonl),
        }
    )
    summary_path = Path(f"{args.output_jsonl}.summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
