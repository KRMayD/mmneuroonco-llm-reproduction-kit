import ast
import csv
import json
import logging
import math
import os
import random
import sys
import braceexpand
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def _load_transformed_image(transforms, image_path):
    with Image.open(str(image_path)) as image:
        return transforms(image)


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        with open(input_filename, "r", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=sep))

        self.images = [row[img_key] for row in rows]
        self.captions = [row[caption_key] for row in rows]
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = _load_transformed_image(self.transforms, self.images[idx])
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class DpoCsvDataset(Dataset):
    def __init__(
            self,
            input_filename,
            transforms,
            img_key,
            img_neg_key,
            caption_key,
            caption_neg_key,
            sep="\t",
            tokenizer=None,
            dpo_mode="standard",
            atomic_attrs=None,
            atomic_train_mode="row_all",
            atomic_pos_suffix="_pos",
            atomic_neg_suffix="_neg",
            atomic_text_weight_mode="uniform",
            semantic_weight_cache_path=None,
            atomic_visual_neg_source="global_neg",
            atomic_visual_attr_json_path=None,
            atomic_visual_match_pool="train_csv",
            atomic_visual_max_other_mismatches=4,
    ):
        logging.debug(f'Loading DPO csv data from {input_filename}.')
        with open(input_filename, "r", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=sep))

        self.input_filename = input_filename
        self.images = [row[img_key] for row in rows]
        self.images_neg = [row[img_neg_key] for row in rows]
        self.captions = [row[caption_key] for row in rows]
        self.captions_neg = [row[caption_neg_key] for row in rows]
        self.transforms = transforms
        self.dpo_mode = dpo_mode
        self.atomic_attrs = atomic_attrs or []
        self.atomic_train_mode = atomic_train_mode
        self.atomic_pos_suffix = atomic_pos_suffix
        self.atomic_neg_suffix = atomic_neg_suffix
        self.atomic_text_weight_mode = atomic_text_weight_mode
        self.semantic_weight_cache_path = semantic_weight_cache_path
        self.atomic_visual_neg_source = atomic_visual_neg_source
        self.atomic_visual_attr_json_path = atomic_visual_attr_json_path
        self.atomic_visual_match_pool = atomic_visual_match_pool
        self.atomic_visual_max_other_mismatches = atomic_visual_max_other_mismatches
        self.atomic_pos_captions = []
        self.atomic_neg_captions = []
        self.atomic_masks = []
        self.atomic_samples = []
        self.atomic_text_weights = None
        self.atomic_visual_neg_paths = None
        self.atomic_visual_neg_found_masks = None
        self.atomic_visual_neg_mismatch_counts = None

        if self.atomic_train_mode not in {"row_all", "prop_sample"}:
            raise ValueError(f"Unsupported atomic_train_mode: {self.atomic_train_mode}")
        if self.atomic_visual_neg_source == "atomic_matched":
            if self.dpo_mode != "atomic":
                raise ValueError("atomic_matched visual negatives require dpo_mode='atomic'")
            if self.atomic_train_mode != "row_all":
                raise ValueError("atomic_matched visual negatives currently require atomic_train_mode='row_all'")

        if self.dpo_mode == "atomic":
            for row_idx, row in enumerate(rows):
                atomic_pos = []
                atomic_neg = []
                atomic_mask = []
                for attr in self.atomic_attrs:
                    pos_key = f"atomic_{attr}{self.atomic_pos_suffix}"
                    neg_key = f"atomic_{attr}{self.atomic_neg_suffix}"
                    pos_val = row.get(pos_key)
                    neg_val = row.get(neg_key)
                    is_valid = (
                        pos_val is not None
                        and neg_val is not None
                        and str(pos_val).strip() != ""
                        and str(neg_val).strip() != ""
                        and str(pos_val).strip().lower() != "nan"
                        and str(neg_val).strip().lower() != "nan"
                    )
                    atomic_pos.append("" if not is_valid else str(pos_val))
                    atomic_neg.append("" if not is_valid else str(neg_val))
                    atomic_mask.append(1 if is_valid else 0)
                self.atomic_pos_captions.append(atomic_pos)
                self.atomic_neg_captions.append(atomic_neg)
                self.atomic_masks.append(atomic_mask)
                if self.atomic_train_mode == "prop_sample":
                    for attr_idx, is_valid in enumerate(atomic_mask):
                        if is_valid:
                            self.atomic_samples.append((row_idx, attr_idx))
            if self.atomic_train_mode == "row_all" and self.atomic_text_weight_mode == "semantic_ref_cache":
                self.atomic_text_weights = self._load_semantic_weight_cache(len(rows))
            if self.atomic_train_mode == "row_all" and self.atomic_visual_neg_source == "atomic_matched":
                self._build_atomic_visual_negatives()
        logging.debug('Done loading DPO data.')

        self.tokenize = tokenizer

    def _load_atomic_visual_metadata(self):
        if not self.atomic_visual_attr_json_path:
            raise ValueError(
                "atomic_matched visual negatives require --atomic-visual-attr-json-path / atomic_visual_attr_json_path"
            )
        with open(self.atomic_visual_attr_json_path, "r") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("atomic visual metadata JSON must be a list of per-image objects")

        metadata_root = os.path.dirname(os.path.abspath(self.atomic_visual_attr_json_path))
        metadata_items = []
        metadata_by_base = {}
        for idx, item in enumerate(payload):
            rel_image_path = item.get("image_path")
            attr_map = item.get("attributes")
            if not rel_image_path or not isinstance(attr_map, dict):
                raise ValueError(
                    f"atomic visual metadata row {idx} must contain image_path and attributes"
                )

            missing_attrs = [attr for attr in self.atomic_attrs if attr not in attr_map]
            if missing_attrs:
                raise ValueError(
                    f"atomic visual metadata row {idx} is missing attrs: {','.join(missing_attrs)}"
                )

            base_name = os.path.basename(rel_image_path)
            if base_name in metadata_by_base:
                raise ValueError(f"Duplicate metadata basename for atomic visual matching: {base_name}")

            full_path = os.path.join(metadata_root, rel_image_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(
                    f"Atomic visual metadata image does not exist: {full_path}"
                )

            meta_item = {
                "base_name": base_name,
                "full_path": full_path,
                "attrs": {attr: str(attr_map[attr]) for attr in self.atomic_attrs},
            }
            metadata_items.append(meta_item)
            metadata_by_base[base_name] = meta_item

        return metadata_items, metadata_by_base

    def _select_atomic_visual_negative(self, row_meta, attr_name, pool_items):
        best_key = None
        best_candidate = None
        row_attrs = row_meta["attrs"]

        for candidate in pool_items:
            if candidate["base_name"] == row_meta["base_name"]:
                continue

            candidate_attrs = candidate["attrs"]
            if candidate_attrs[attr_name] == row_attrs[attr_name]:
                continue

            other_mismatches = 0
            for other_attr in self.atomic_attrs:
                if other_attr == attr_name:
                    continue
                if candidate_attrs[other_attr] != row_attrs[other_attr]:
                    other_mismatches += 1

            if other_mismatches > self.atomic_visual_max_other_mismatches:
                continue

            sort_key = (other_mismatches, candidate["full_path"])
            if best_key is None or sort_key < best_key:
                best_key = sort_key
                best_candidate = {
                    "full_path": candidate["full_path"],
                    "other_mismatches": other_mismatches,
                }

        return best_candidate

    def _build_atomic_visual_negatives(self):
        metadata_items, metadata_by_base = self._load_atomic_visual_metadata()
        row_metas = []
        for row_idx, image_path in enumerate(self.images):
            base_name = os.path.basename(image_path)
            row_meta = metadata_by_base.get(base_name)
            if row_meta is None:
                raise ValueError(
                    f"Could not find atomic visual metadata for CSV row {row_idx} basename {base_name}"
                )
            row_metas.append(row_meta)

        if self.atomic_visual_match_pool == "train_csv":
            pool_base_names = sorted({row_meta["base_name"] for row_meta in row_metas})
            pool_items = [metadata_by_base[base_name] for base_name in pool_base_names]
        elif self.atomic_visual_match_pool == "all_metadata":
            pool_items = metadata_items
        else:
            raise ValueError(
                f"Unsupported atomic_visual_match_pool: {self.atomic_visual_match_pool}"
            )

        self.atomic_visual_neg_paths = []
        self.atomic_visual_neg_found_masks = []
        self.atomic_visual_neg_mismatch_counts = []
        bucket_counts = Counter()
        valid_total = 0

        for row_idx, row_meta in enumerate(row_metas):
            row_neg_paths = []
            row_found_mask = []
            row_mismatch_counts = []
            fallback_path = self.images_neg[row_idx]

            for attr_idx, attr_name in enumerate(self.atomic_attrs):
                if not self.atomic_masks[row_idx][attr_idx]:
                    row_neg_paths.append(fallback_path)
                    row_found_mask.append(False)
                    row_mismatch_counts.append(self.atomic_visual_max_other_mismatches + 1)
                    continue

                valid_total += 1
                candidate = self._select_atomic_visual_negative(
                    row_meta=row_meta,
                    attr_name=attr_name,
                    pool_items=pool_items,
                )
                if candidate is None:
                    row_neg_paths.append(fallback_path)
                    row_found_mask.append(True)
                    row_mismatch_counts.append(self.atomic_visual_max_other_mismatches + 1)
                    bucket_counts["fallback"] += 1
                else:
                    row_neg_paths.append(candidate["full_path"])
                    row_found_mask.append(True)
                    row_mismatch_counts.append(candidate["other_mismatches"])
                    bucket_counts[candidate["other_mismatches"]] += 1

            self.atomic_visual_neg_paths.append(row_neg_paths)
            self.atomic_visual_neg_found_masks.append(row_found_mask)
            self.atomic_visual_neg_mismatch_counts.append(row_mismatch_counts)

        ordered_bucket_counts = {
            mismatch: int(bucket_counts.get(mismatch, 0))
            for mismatch in range(self.atomic_visual_max_other_mismatches + 1)
        }
        ordered_bucket_counts["fallback"] = int(bucket_counts.get("fallback", 0))
        logging.info(
            "Atomic matched visual negatives initialized: valid_pairs=%d pool=%s buckets=%s",
            valid_total,
            self.atomic_visual_match_pool,
            ordered_bucket_counts,
        )

    def _load_semantic_weight_cache(self, num_rows):
        if not self.semantic_weight_cache_path:
            raise ValueError(
                "semantic_ref_cache mode requires --semantic-weight-cache-path / semantic_weight_cache_path"
            )
        with open(self.semantic_weight_cache_path, "r") as handle:
            payload = json.load(handle)

        expected_csv = os.path.abspath(self.input_filename)
        cached_csv = os.path.abspath(payload.get("csv_path", ""))
        if expected_csv != cached_csv:
            raise ValueError(
                f"semantic cache csv mismatch: expected {expected_csv}, got {cached_csv}"
            )
        if payload.get("num_rows") != num_rows:
            raise ValueError(
                f"semantic cache row count mismatch: expected {num_rows}, got {payload.get('num_rows')}"
            )
        if payload.get("atomic_attrs") != self.atomic_attrs:
            raise ValueError(
                f"semantic cache atomic_attrs mismatch: expected {self.atomic_attrs}, got {payload.get('atomic_attrs')}"
            )
        if payload.get("atomic_pos_suffix") != self.atomic_pos_suffix:
            raise ValueError(
                "semantic cache atomic_pos_suffix mismatch: "
                f"expected {self.atomic_pos_suffix}, got {payload.get('atomic_pos_suffix')}"
            )
        if payload.get("atomic_neg_suffix") != self.atomic_neg_suffix:
            raise ValueError(
                "semantic cache atomic_neg_suffix mismatch: "
                f"expected {self.atomic_neg_suffix}, got {payload.get('atomic_neg_suffix')}"
            )

        semantic_weight = payload.get("semantic_weight")
        valid_mask = payload.get("valid_mask")
        if semantic_weight is None or valid_mask is None:
            raise ValueError("semantic cache must contain semantic_weight and valid_mask")
        if len(semantic_weight) != num_rows or len(valid_mask) != num_rows:
            raise ValueError("semantic cache row-aligned arrays must match dataframe length")

        row_weights = []
        for row_idx, (weights_row, valid_row, dataset_mask) in enumerate(
                zip(semantic_weight, valid_mask, self.atomic_masks)
        ):
            if len(weights_row) != len(self.atomic_attrs) or len(valid_row) != len(self.atomic_attrs):
                raise ValueError(
                    f"semantic cache row {row_idx} does not match atomic attr count {len(self.atomic_attrs)}"
                )
            cache_mask = [1 if bool(x) else 0 for x in valid_row]
            if cache_mask != dataset_mask:
                raise ValueError(
                    f"semantic cache valid_mask mismatch at row {row_idx}: expected {dataset_mask}, got {cache_mask}"
                )
            row_weights.append([
                float(weight) if cache_mask[attr_idx] else 0.0
                for attr_idx, weight in enumerate(weights_row)
            ])
        return row_weights

    def __len__(self):
        if self.dpo_mode == "atomic" and self.atomic_train_mode == "prop_sample":
            return len(self.atomic_samples)
        return len(self.captions)

    def __getitem__(self, idx):
        if self.dpo_mode != "atomic":
            images = _load_transformed_image(self.transforms, self.images[idx])
            images_neg = _load_transformed_image(self.transforms, self.images_neg[idx])
            texts = self.tokenize([str(self.captions[idx])])[0]
            texts_neg = self.tokenize([str(self.captions_neg[idx])])[0]
            return images, images_neg, texts, texts_neg

        if self.atomic_train_mode == "prop_sample":
            row_idx, attr_idx = self.atomic_samples[idx]
            images = _load_transformed_image(self.transforms, self.images[row_idx])
            images_neg = _load_transformed_image(self.transforms, self.images_neg[row_idx])
            texts = self.tokenize([str(self.captions[row_idx])])[0]
            texts_neg = self.tokenize([str(self.captions_neg[row_idx])])[0]
            atomic_text_pos = self.tokenize([self.atomic_pos_captions[row_idx][attr_idx]])[0]
            atomic_text_neg = self.tokenize([self.atomic_neg_captions[row_idx][attr_idx]])[0]
            atomic_attr_idx = torch.tensor(attr_idx, dtype=torch.long)
            atomic_mask = torch.tensor(True, dtype=torch.bool)
            return (
                images,
                images_neg,
                texts,
                texts_neg,
                atomic_text_pos,
                atomic_text_neg,
                atomic_attr_idx,
                atomic_mask,
            )

        images = _load_transformed_image(self.transforms, self.images[idx])
        images_neg = _load_transformed_image(self.transforms, self.images_neg[idx])
        texts = self.tokenize([str(self.captions[idx])])[0]
        texts_neg = self.tokenize([str(self.captions_neg[idx])])[0]
        atomic_texts_pos = self.tokenize(self.atomic_pos_captions[idx])
        atomic_texts_neg = self.tokenize(self.atomic_neg_captions[idx])
        atomic_mask = torch.tensor(self.atomic_masks[idx], dtype=torch.bool)
        payload = (
            images,
            images_neg,
            texts,
            texts_neg,
            atomic_texts_pos,
            atomic_texts_neg,
            atomic_mask,
        )
        if self.atomic_text_weights is not None:
            atomic_text_weight = torch.tensor(self.atomic_text_weights[idx], dtype=torch.float32)
            payload = payload + (atomic_text_weight,)
        if self.atomic_visual_neg_source == "atomic_matched":
            atomic_neg_images = torch.stack([
                _load_transformed_image(self.transforms, image_path)
                for image_path in self.atomic_visual_neg_paths[idx]
            ], dim=0)
            atomic_neg_found_mask = torch.tensor(
                self.atomic_visual_neg_found_masks[idx],
                dtype=torch.bool,
            )
            atomic_neg_mismatch_count = torch.tensor(
                self.atomic_visual_neg_mismatch_counts[idx],
                dtype=torch.long,
            )
            payload = payload + (
                atomic_neg_images,
                atomic_neg_found_mask,
                atomic_neg_mismatch_count,
            )
        return payload


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    
    if args.dpo_loss and is_train:
        dataset = DpoCsvDataset(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            img_neg_key=args.csv_img_neg_key,
            caption_key=args.csv_caption_key,
            caption_neg_key=args.csv_caption_neg_key,
            sep=args.csv_separator,
            tokenizer=tokenizer,
            dpo_mode=args.dpo_mode,
            atomic_attrs=args.atomic_attrs,
            atomic_train_mode=args.atomic_train_mode,
            atomic_pos_suffix=args.atomic_pos_suffix,
            atomic_neg_suffix=args.atomic_neg_suffix,
            atomic_text_weight_mode=getattr(args, "atomic_text_weight_mode", "uniform"),
            semantic_weight_cache_path=getattr(args, "semantic_weight_cache_path", None),
            atomic_visual_neg_source=getattr(args, "atomic_visual_neg_source", "global_neg"),
            atomic_visual_attr_json_path=getattr(args, "atomic_visual_attr_json_path", None),
            atomic_visual_match_pool=getattr(args, "atomic_visual_match_pool", "train_csv"),
            atomic_visual_max_other_mismatches=getattr(args, "atomic_visual_max_other_mismatches", 4),
        )
    else:
        dataset = CsvDataset(
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            tokenizer=tokenizer
        )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(
            self,
            transform=None,
            image_size=(224, 224),
            caption="Dummy caption",
            dataset_size=100,
            tokenizer=None,
    ):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
