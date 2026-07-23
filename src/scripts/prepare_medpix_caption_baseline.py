#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path


DEFAULT_INSTRUCTION = "Please describe the medical image in English."


def build_record(row_id: int, image_path: str, caption: str, instruction: str):
    return {
        "id": f"medpix::{row_id}",
        "image": [image_path],
        "conversations": [
            {"from": "human", "value": f"{instruction}\n<image>"},
            {"from": "gpt", "value": caption},
        ],
        "source": "MedPix",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    args = parser.parse_args()

    rows = []
    with args.csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            image_path = (row.get("filename") or "").strip()
            caption = " ".join((row.get("Caption") or "").strip().split())
            if not image_path or not caption:
                continue
            if not Path(image_path).is_file():
                continue
            rows.append(build_record(idx, image_path, caption, args.instruction))

    required = args.train_size + args.test_size
    if len(rows) < required:
        raise ValueError(f"Need at least {required} valid rows, found {len(rows)}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    train_rows = rows[: args.train_size]
    test_rows = rows[args.train_size : args.train_size + args.test_size]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train_data.json"
    test_path = args.output_dir / "test_data.json"

    train_path.write_text(json.dumps(train_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    test_path.write_text(json.dumps(test_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "csv_path": str(args.csv_path),
        "output_dir": str(args.output_dir),
        "instruction": args.instruction,
        "valid_rows": len(rows),
        "train_size": len(train_rows),
        "test_size": len(test_rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
