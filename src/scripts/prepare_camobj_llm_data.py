#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def convert_row(row, dataset_root: Path, source: str):
    images = row.get('images') or []
    if not images:
        return None
    abs_images = []
    for img in images:
        p = dataset_root / img
        if not p.exists() and img.startswith('data/'):
            p = dataset_root / img[len('data/'): ]
        if not p.exists():
            return None
        abs_images.append(str(p.resolve()))

    messages = row.get('messages') or []
    if len(messages) < 2:
        return None
    user = messages[0].get('content', '').strip()
    assistant = messages[1].get('content', '').strip()
    if not user or not assistant:
        return None
    if '<image>' not in user:
        user = user + '\n<image>'
    return {
        'image': abs_images,
        'source': source,
        'conversations': [
            {'from': 'human', 'value': user},
            {'from': 'gpt', 'value': assistant},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mmcamobj-root', default='/home/msko021220/MM-CamObj')
    ap.add_argument('--output-dir', default='/home/msko021220/project/finegrainedVLM-vision-encoder-llm/data/camobj_official_grouped_seed42')
    ap.add_argument('--val-ratio', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    mm_root = Path(args.mmcamobj_root)
    dataset_root = mm_root / 'dataset'
    train_data_dir = dataset_root / 'train_data'
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in ['camobj_align.json', 'camobj_instruct.json']:
        src_rows = load_json(train_data_dir / name)
        source = name.replace('.json', '')
        for r in src_rows:
            item = convert_row(r, dataset_root, source)
            if item is not None:
                rows.append(item)

    # camobj_align and camobj_instruct contain different targets for the same
    # image. Splitting individual rows would leak the image into validation.
    rows_by_image = {}
    for row in rows:
        image_key = tuple(row['image'])
        rows_by_image.setdefault(image_key, []).append(row)

    image_groups = list(rows_by_image)
    rng = random.Random(args.seed)
    rng.shuffle(image_groups)
    n_val_groups = max(1, int(round(len(image_groups) * args.val_ratio)))
    val_image_groups = set(image_groups[:n_val_groups])
    train_rows = []
    val_rows = []
    for image_key, grouped_rows in rows_by_image.items():
        destination = val_rows if image_key in val_image_groups else train_rows
        destination.extend(grouped_rows)

    benchmark_images = set()
    for question_name in ['easy_vqa.jsonl', 'hard_vqa.jsonl']:
        question_path = dataset_root / 'questions' / question_name
        with question_path.open('r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                question = json.loads(line)
                image = dataset_root / question['image'].removeprefix('data/')
                benchmark_images.add(str(image.resolve()))
    converted_images = {row['image'][0] for row in rows}
    benchmark_overlap = sorted(converted_images & benchmark_images)
    if benchmark_overlap:
        raise ValueError(
            'CamObj instruction data overlaps Easy/Hard benchmark images: '
            f'{benchmark_overlap[:5]}'
        )

    (out / 'train_data.json').write_text(json.dumps(train_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'test_data.json').write_text(json.dumps(val_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {
        'mmcamobj_root': str(mm_root),
        'dataset_root': str(dataset_root),
        'output_dir': str(out),
        'total_converted': len(rows),
        'total_unique_images': len(image_groups),
        'train_samples': len(train_rows),
        'val_samples': len(val_rows),
        'train_unique_images': len({row['image'][0] for row in train_rows}),
        'val_unique_images': len({row['image'][0] for row in val_rows}),
        'train_val_image_overlap': len(
            {row['image'][0] for row in train_rows} & {row['image'][0] for row in val_rows}
        ),
        'easy_hard_benchmark_image_overlap': len(benchmark_overlap),
        'split_unit': 'image_path',
        'val_ratio': args.val_ratio,
        'seed': args.seed,
        'sources': ['camobj_align.json', 'camobj_instruct.json'],
    }
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
