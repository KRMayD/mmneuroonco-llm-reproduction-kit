import torch
import json
import os
import re
import random

from typing import Dict, Sequence
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler


def get_sample_images(data_dict):
    images = data_dict.get("image", data_dict.get("images"))
    if isinstance(images, str):
        return [images]
    return images or []


def get_sample_instruction_and_target(data_dict):
    conversations = data_dict.get("conversations")
    if isinstance(conversations, list) and len(conversations) > 1:
        human = conversations[0]
        assistant = conversations[1]
        if human.get("from") == "human" and assistant.get("from") == "gpt":
            return human.get("value", ""), assistant.get("value", "")

    messages = data_dict.get("messages")
    if isinstance(messages, list) and len(messages) > 1:
        user = messages[0]
        assistant = messages[1]
        if user.get("role") == "user" and assistant.get("role") == "assistant":
            return user.get("content", ""), assistant.get("content", "")

    raise ValueError("Expected a human/user and gpt/assistant message pair.")


# split text caption
def split_caption(text):
    """Split captions by sentence-ending markers."""
    return [cap.strip() for cap in re.split(r'\n|</s>|[.]', text) if cap.strip()]

def random_sample_from_list(captions_list, k, merged_num=1):
    n = len(captions_list)
    if merged_num == 1:
        if n >= k:
            return random.sample(captions_list, k)
        else:  #minimizing caption dupilications
            return random.choices(captions_list, k=k)
            #return captions_list + random.sample(captions_list, k - n)
    elif merged_num >= n:
        return ['. '.join(captions_list)]
    else:
        sampled_list = []
        sampled_indices = draw_numbers(n=n - merged_num, k=k)
        for sampled_index in sampled_indices:
            sampled_list.append('. '.join(captions_list[sampled_index:sampled_index + merged_num]))
        return sampled_list

def draw_numbers(n, k=4):
    population = list(range(0, n))
    if n >= k:
        return random.sample(population, k)
    else:
        return random.choices(population, k=k)

def sample_dict(text, k=3, tokenizer=None, sampling_mode='diverse_sampling', pixelprose=True, max_merged_num=3, return_text=False):

    if sampling_mode == 'diverse_sampling':
        if pixelprose:
            # raw_caption = text["caption"]
            # captions_list = split_caption(raw_caption)
            if isinstance(text, list):
                text = text[0]
            captions_list = split_caption(text)
        else:
            captions_list = (text['raw_caption'] + text['shortIB_captions'] + text['longIB_captions'] +
                             text['shortSV_captions'] + text['longSV_captions'] +
                             text['shortLLA_captions'] + text['longLLA_captions'])
        n_captions = len(captions_list)
        sampled_sentences = []
        for _ in range(k):
            merged_num = random.randint(1, max_merged_num)
            if merged_num == 1:
                # Sample one caption
                sampled_sentence = random.choice(captions_list)
                sampled_sentences.append(sampled_sentence)
            else:
                prob_flag = 0.5 # 50% merging subsequent captions, 50% merging captions from random positions
                if random.random() < prob_flag:
                    sampled_sentence_list = random_sample_from_list(
                        captions_list, k=1, merged_num=merged_num)
                    sampled_sentences.extend(sampled_sentence_list)
                else:
                    # Randomly select captions to merge
                    if n_captions >= merged_num:
                        captions_to_merge = random.sample(captions_list, merged_num)
                    else:
                        captions_to_merge = [random.choice(captions_list) for _ in range(merged_num)]
                    # Merge the captions
                    sampled_sentence = '. '.join(captions_to_merge)
                    sampled_sentences.append(sampled_sentence)

        # tokenized_sentences = tokenizer(sampled_sentences)
        if return_text:
            return sampled_sentences

        tokenized_sentences = tokenizer(sampled_sentences, 
                                        padding='max_length', 
                                        max_length=256, 
                                        truncation=True, 
                                        return_tensors='pt')
        
        return tokenized_sentences
    else:
        raise NotImplementedError('Please select a valid sampling method')

def default_collate_fn(instances: Sequence[Dict]):
    # check instances
    if not instances:
        raise ValueError("Input instances list is empty")

    text = []
    split_text = []
    instructions = []
    path_features = []
    raw_image_paths = []

    for instance in instances:
        text.append(instance["text"])
        split_text.append(instance["split_text"])
        instructions.append(instance.get("instruction", ""))
        if "path" in instance:
            path_features.append(instance["path"])
        if "raw_image_paths" in instance:
            raw_image_paths.append(instance["raw_image_paths"])

    has_path = len(path_features) == len(instances)
    has_raw = len(raw_image_paths) == len(instances)
    if has_path and has_raw:
        raise ValueError("A batch cannot contain both precomputed features and raw image paths.")
    if not has_path and not has_raw:
        raise ValueError("Expected each instance to contain either 'path' or 'raw_image_paths'.")

    data_dict = {
        "text": text,
        "split_text": split_text,
        "instruction": instructions,
    }

    if has_path:
        normalized_features = []
        max_tokens = 0
        for path_feature in path_features:
            if path_feature.dim() == 1:
                path_feature = path_feature.unsqueeze(0)
            elif path_feature.dim() != 2:
                raise ValueError(
                    "Expected each precomputed feature tensor to have shape [D] or [T, D], "
                    f"but got {tuple(path_feature.shape)}."
                )
            normalized_features.append(path_feature)
            max_tokens = max(max_tokens, path_feature.size(0))

        padded_features = []
        path_masks = []
        for path_feature in normalized_features:
            num_tokens = path_feature.size(0)
            if num_tokens < max_tokens:
                pad = torch.zeros(
                    (max_tokens - num_tokens, path_feature.size(-1)),
                    dtype=path_feature.dtype,
                )
                path_feature = torch.cat([path_feature, pad], dim=0)
            padded_features.append(path_feature)

            path_mask = torch.zeros(max_tokens, dtype=torch.long)
            path_mask[:num_tokens] = 1
            path_masks.append(path_mask)

        data_dict["path"] = torch.stack(padded_features)
        data_dict["path_mask"] = torch.stack(path_masks)
    else:
        data_dict["raw_image_paths"] = raw_image_paths

    return data_dict

# Todo
def collate_fn_SW(instances: Sequence[Dict]):
    return default_collate_fn(instances)


class PathFLIP_Dataset_SW(Dataset):

    def __init__(self,
        data_path=None,
        path_sample=False,
        slide_window_size=256,
        path_sample_windows_num=256, # 256 x 256 = 65536
        max_dataset_length=None,
        tokenizer=None,
        text_sample_num=3,
        sampling_mode='diverse_sampling',
        max_merged_num=3,
        ):
        super().__init__()
    
        self.path_sample = path_sample
        self.slide_window_size = slide_window_size
        self.path_sample_windows_num = path_sample_windows_num
        # self.path_sample_num = path_sample_num
        self.tokenizer = tokenizer
        self.sampling_mode = sampling_mode
        self.max_merged_num = max_merged_num
        self.text_sample_num = text_sample_num

        if data_path.endswith('.json'):
            json_data = json.load(open(data_path))
        else:
            raise NotImplementedError

        # check json data
        valid_json_data = []
        for item in json_data:
            image_list = get_sample_images(item)
            if image_list:
                feature_path = image_list[0]
                if os.path.exists(feature_path):
                    valid_json_data.append(item)
        json_data = valid_json_data

        if max_dataset_length is not None and len(json_data)>max_dataset_length:
            json_data = json_data[:max_dataset_length]
        
        self.json_data = json_data

        
    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        data_dict = self.json_data[index]
        data_return = {'text': None}

        instruction, target = get_sample_instruction_and_target(data_dict)
        data_dict['text'] = target
        data_return['instruction'] = instruction
        
        if self.tokenizer is not None:

            data_return['text'] = self.tokenizer(data_dict['text'], 
                                                truncation=True, 
                                                padding='max_length', 
                                                max_length=512,
                                                return_tensors="pt")
            
            data_return['split_text'] = sample_dict(text=data_dict['text'],
                                                k=self.text_sample_num,
                                                tokenizer=self.tokenizer,
                                                sampling_mode=self.sampling_mode,
                                                max_merged_num=self.max_merged_num)
        else:
            data_return['text'] = data_dict['text']
            
            data_return['split_text'] = sample_dict(text=data_dict['text'],
                                                k=self.text_sample_num,
                                                sampling_mode=self.sampling_mode,
                                                max_merged_num=self.max_merged_num,
                                                return_text=True)

        image_list = get_sample_images(data_dict)
        if isinstance(image_list, str):
            image_list = [image_list]

        pt_files = [image_file for image_file in image_list if image_file.endswith(".pt")]
        raw_image_files = [image_file for image_file in image_list if not image_file.endswith(".pt")]

        if pt_files and raw_image_files:
            raise ValueError(
                "Each sample must use either precomputed .pt features or raw image paths, not both."
            )

        if pt_files:
            patch_features = []
            for image_file in pt_files:
                slide_feature = torch.load(image_file, weights_only=True)
                if slide_feature.dim() == 1:
                    slide_feature = slide_feature.unsqueeze(0)
                elif slide_feature.dim() > 2:
                    slide_feature = slide_feature.reshape(-1, slide_feature.shape[-1])
                patch_features.append(slide_feature)

            patch_features = torch.cat(patch_features, dim=0)
            data_return["path"] = patch_features
            data_return["path_mask"] = torch.ones(patch_features.shape[0], dtype=torch.long)
        else:
            data_return["raw_image_paths"] = image_list

        return data_return


class pathVL_Dataset_dm(LightningDataModule):
    def __init__(
        self,
        data_path=None,
        path_sample=True,
        slide_window_size=256,
        path_sample_windows_num=256,
        tokenizer=None,
        max_dataset_length=None,
        batch_size: int = 4,
        num_workers: int = 4,
        args=None,
    ):
        super().__init__()
        self.path_sample = path_sample
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args

        self.train_dataset = PathFLIP_Dataset_SW(
            data_path=os.path.join(data_path, 'train_data.json'),
            path_sample=path_sample,
            slide_window_size=slide_window_size,
            path_sample_windows_num=path_sample_windows_num,
            max_dataset_length=max_dataset_length,
            tokenizer=tokenizer,
            text_sample_num=args.text_sample_num,
            sampling_mode=args.sampling_mode,
            max_merged_num=args.max_merged_num,
        )

        self.val_dataset = None
        if not (args.mode == 'train' and args.skip_validation):
            self.val_dataset = PathFLIP_Dataset_SW(
                data_path=os.path.join(data_path, 'test_data.json'),
                path_sample=path_sample,
                slide_window_size=slide_window_size,
                path_sample_windows_num=path_sample_windows_num,
                max_dataset_length=max_dataset_length,
                tokenizer=tokenizer,
                text_sample_num=args.text_sample_num,
                sampling_mode=args.sampling_mode,
                max_merged_num=args.max_merged_num,
            )
        # self.test_dataset = None

        self.collate_fn = default_collate_fn
    
    def train_dataloader(self):

        # 判断是否是分布式训练
        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(self.train_dataset)
            shuffle = False  # 分布式 sampler 内部 shuffle，不需要 DataLoader 再 shuffle
        else:
            sampler = None
            shuffle = True

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.num_workers > 0),
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self.collate_fn,
        )
        
        return train_loader

    def val_dataloader(self):

        if self.val_dataset is None:
            return None

        if self.trainer and self.trainer.world_size > 1:
            sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            sampler = None

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.num_workers > 0),
            sampler=sampler,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        return val_loader


# Test
if __name__ == '__main__':
    from utils.process_args import get_args
    args = get_args()
    dm = pathVL_Dataset_dm(
        data_path=args.data_path,
        slide_window_size=args.slide_window_size,
        path_sample_windows_num=args.path_sample_windows_num,
        max_dataset_length=args.max_dataset_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        args=args
    )
    data = dm.train_dataloader().dataset[0]
    print(data)
