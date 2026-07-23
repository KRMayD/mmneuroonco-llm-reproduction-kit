#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path


TRAIN_VARIANTS = {
    "open_closed_nocot": ("train_open.jsonl", "train_no_cot_closed.jsonl"),
    "open_closed_cot": ("train_open.jsonl", "train_cot_closed.jsonl"),
    "open_closed_both": (
        "train_open.jsonl",
        "train_no_cot_closed.jsonl",
        "train_cot_closed.jsonl",
    ),
    "open_only": ("train_open.jsonl",),
    "closed_nocot_only": ("train_no_cot_closed.jsonl",),
    "closed_cot_only": ("train_cot_closed.jsonl",),
}


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def select_extracted_root(root, name):
    base = root / "images" / name
    nested = base / name
    return nested if nested.is_dir() else base


def resolve_image_path(root, relative_path):
    relative_path = str(relative_path).strip()
    direct = root / relative_path
    if direct.exists():
        return str(direct.resolve())

    stem = direct.with_suffix("")
    for extension in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = stem.with_suffix(extension)
        if candidate.exists():
            return str(candidate.resolve())
    return None


def role_to_conversation(messages):
    if len(messages) < 2:
        raise ValueError("Expected at least two messages.")
    user, assistant = messages[0], messages[1]
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError("Unexpected message roles.")
    return str(user.get("content", "")), str(assistant.get("content", ""))


def convert_train_file(path, image_root, source_name, limit=0):
    converted = []
    stats = {
        "source": source_name,
        "rows": 0,
        "written": 0,
        "missing_images": 0,
        "bad_rows": 0,
        "question_type": Counter(),
    }
    for row in iter_jsonl(path):
        stats["rows"] += 1
        if limit and stats["rows"] > limit:
            break
        try:
            relative_images = row.get("images") or []
            if not relative_images:
                raise ValueError("Missing images.")
            resolved = [resolve_image_path(image_root, item) for item in relative_images]
            if any(item is None for item in resolved):
                stats["missing_images"] += 1
                continue
            instruction, target = role_to_conversation(row.get("messages") or [])
            meta = dict(row.get("meta") or {})
            meta.update({"source": source_name, "relative_images": relative_images})
            stats["question_type"][str(meta.get("question_type", "unknown"))] += 1
            converted.append(
                {
                    "image": resolved,
                    "conversations": [
                        {"from": "human", "value": instruction},
                        {"from": "gpt", "value": target},
                    ],
                    "meta": meta,
                }
            )
            stats["written"] += 1
        except Exception:
            stats["bad_rows"] += 1
    stats["question_type"] = dict(stats["question_type"])
    return converted, stats


def build_options_text(options):
    return "\n".join(
        f"{key}. {options[key]}" for key in ("A", "B", "C", "D", "E") if key in options
    )


def convert_closed_benchmark(path, image_root):
    data = read_json(path)
    converted = []
    stats = {
        "source": "MM-NeuroOnco_Benchmark_Closed",
        "images": len(data),
        "pairs": 0,
        "written": 0,
        "missing_images": 0,
        "question_type": Counter(),
    }
    for image_item in data:
        annotated_path = str(image_item.get("image_path", ""))
        local_path = resolve_image_path(image_root, annotated_path)
        pairs = image_item.get("vqa_pairs") or []
        if local_path is None:
            stats["missing_images"] += len(pairs)
            continue
        for question_index, qa in enumerate(pairs):
            stats["pairs"] += 1
            question = str(qa.get("question", ""))
            options = qa.get("options") or {}
            answer = str(qa.get("answer", "")).strip().upper()
            instruction = (
                "<image>\n"
                f"{question}\n\n"
                "Options:\n"
                f"{build_options_text(options)}\n\n"
                "Answer with the option letter only."
            )
            question_type = str(qa.get("question_type", "unknown"))
            stats["question_type"][question_type] += 1
            converted.append(
                {
                    "image": [local_path],
                    "conversations": [
                        {"from": "human", "value": instruction},
                        {"from": "gpt", "value": answer},
                    ],
                    "meta": {
                        "source": stats["source"],
                        "id": image_item.get("id"),
                        "q_index": question_index,
                        "question_type": question_type,
                        "question": question,
                        "options": options,
                        "answer": answer,
                        "image_path": annotated_path,
                    },
                }
            )
            stats["written"] += 1
    stats["question_type"] = dict(stats["question_type"])
    return converted, stats


def convert_open_benchmark(path, image_root):
    data = read_json(path)
    converted = []
    stats = {
        "source": "MM-NeuroOnco_Benchmark_Open",
        "images": len(data),
        "pairs": 0,
        "written": 0,
        "missing_images": 0,
        "question_type": Counter(),
    }
    for image_item in data:
        annotated_path = str(image_item.get("image_path", ""))
        local_path = resolve_image_path(image_root, annotated_path)
        pairs = image_item.get("qa_pairs") or image_item.get("vqa_pairs") or []
        if local_path is None:
            stats["missing_images"] += len(pairs)
            continue
        for question_index, qa in enumerate(pairs):
            stats["pairs"] += 1
            question = str(qa.get("question", ""))
            target = str(qa.get("reference_answer", qa.get("answer", "")))
            question_type = str(qa.get("question_type", "unknown"))
            instruction = (
                "<image>\n"
                "Answer the following brain MRI question based on the visible image findings.\n"
                f"Question: {question}"
            )
            stats["question_type"][question_type] += 1
            converted.append(
                {
                    "image": [local_path],
                    "conversations": [
                        {"from": "human", "value": instruction},
                        {"from": "gpt", "value": target},
                    ],
                    "meta": {
                        "source": stats["source"],
                        "id": image_item.get("id"),
                        "q_index": question_index,
                        "question_id": qa.get("question_id"),
                        "question_type": question_type,
                        "question": question,
                        "reference_answer": target,
                        "eval_keywords": qa.get("eval_keywords") or [],
                        "image_path": annotated_path,
                    },
                }
            )
            stats["written"] += 1
    stats["question_type"] = dict(stats["question_type"])
    return converted, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mmneuro-images-root",
        type=Path,
        default=Path("/home/msko021220/dataset/gfnnnb_MM-NeuroOnco-Images"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/msko021220/project/finegrainedVLM-vision-encoder-llm/data/mm_neuroonco_official"
        ),
    )
    parser.add_argument("--train-variant", choices=sorted(TRAIN_VARIANTS), default="open_closed_nocot")
    parser.add_argument("--train-limit-per-file", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.mmneuro_images_root.resolve()
    training_root = select_extracted_root(root, "Dataset")
    benchmark_root = select_extracted_root(root, "Benchmark_Images")
    training_dir = root / "training"
    benchmark_dir = root / "Benchmark"

    required = [training_root, benchmark_root, training_dir, benchmark_dir]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    train_rows = []
    train_stats = []
    for filename in TRAIN_VARIANTS[args.train_variant]:
        rows, stats = convert_train_file(
            training_dir / filename,
            training_root,
            filename,
            args.train_limit_per_file,
        )
        train_rows.extend(rows)
        train_stats.append(stats)
    random.Random(args.seed).shuffle(train_rows)

    closed_rows, closed_stats = convert_closed_benchmark(
        benchmark_dir / "Benchmark_VQA_Closed.json", benchmark_root
    )
    open_rows, open_stats = convert_open_benchmark(
        benchmark_dir / "Benchmark_VQA_Open.json", benchmark_root
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "train_data.json", train_rows)
    write_json(args.output_dir / "test_data.json", closed_rows + open_rows)
    write_json(args.output_dir / "closed_test_data.json", closed_rows)
    write_json(args.output_dir / "open_test_data.json", open_rows)

    stats = {
        "train_variant": args.train_variant,
        "mmneuro_images_root": str(root),
        "training_image_root": str(training_root),
        "benchmark_image_root": str(benchmark_root),
        "train_total": len(train_rows),
        "test_total": len(closed_rows) + len(open_rows),
        "closed_total": len(closed_rows),
        "open_total": len(open_rows),
        "train_files": train_stats,
        "closed_benchmark": closed_stats,
        "open_benchmark": open_stats,
    }
    write_json(args.output_dir / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
