# encoding=utf-8
import os
# for huggingface/tokenizers to work in data loader with multiple workers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import random
from typing import Dict, Iterable, List, Union
import numpy as np
import pandas as pd
from PIL import Image
from types import *
import os.path
from dataclasses import dataclass
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torchvision import transforms
from tqdm import tqdm, trange
import torch
import warnings

@dataclass
class MetaData:
    domain_mapping: Dict[str, int]
    label_mapping: Dict[str, int]
    rgb_mean: List[float]
    rgb_std: List[float]
    max_chars: int = 400
    max_tokens: int = 400

    def __str__(self):
        return str(self.__dict__)

    def reverse_domain_mapping(self):
        return {self.domain_mapping[name]: name for name in self.domain_mapping}

    def reverse_label_mapping(self):
        return {self.label_mapping[name]: name for name in self.label_mapping}

    def save(self, metadata_path: str):
        serialized = json.dumps(
            self, 
            default=lambda o: o.__dict__, 
            sort_keys=True, 
            indent=4
        )
        with open(metadata_path, 'w') as json_file:
            json_file.write(serialized)

    @staticmethod
    def load(metadata_path: str):
        with open(metadata_path) as json_file:
            metadata: dict = json.load(json_file)

        return MetaData(**metadata)

def _compute_images_mean_std(df: pd.DataFrame, images_root_path: str):
    df['image_path'] = df['image_path'].apply(
        lambda image_path: os.path.abspath(os.path.join(images_root_path, image_path))
    )

    image_paths: List[str] = df['image_path'].tolist()

    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # the entire dataset may not fit in memory (~50 gigabytes) so only every 
    # nth image will be used
    select_every_nth = 20

    images = []
    for i in trange(0, len(image_paths), select_every_nth, desc='(Metadata) Mean and std for RGB values'):

        image_path = image_paths[i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            image = Image.open(image_path).convert('RGB')
            image = img_transform(image).unsqueeze(0).numpy()

        images.append(image)

    arr = np.concatenate(images, axis=0)

    mean = np.mean(arr, axis=(0, 2, 3))
    std = np.std(arr, axis=(0, 2, 3))

    return mean.tolist(), std.tolist()
    

def create_metadata(dataset_path: str, images_root_path: str, metadata_path: str, 
    force=False) -> MetaData:

    if not force and os.path.exists(metadata_path):
        return MetaData.load(metadata_path)

    df = pd.read_csv(dataset_path, delimiter='\t')

    topics: List[str] = list(df['topic'].unique())
    topics.sort()
    domain_mapping = {topic: index for index, topic in enumerate(topics)}

    labels: List[str] = list(df['sarcastic'].unique())
    labels.sort()
    label_mapping = {label: index for index, label in enumerate(labels)}

    rgb_mean, rgb_std = _compute_images_mean_std(df, images_root_path)

    metadata = MetaData(
        domain_mapping=domain_mapping, 
        label_mapping=label_mapping,
        rgb_mean=rgb_mean,
        rgb_std=rgb_std
    )
    metadata.save(metadata_path)
    return metadata

def create_image_transform(metadata: MetaData):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(metadata.rgb_mean, metadata.rgb_std)
    ])

def bert_tokenize(texts: Iterable[str], max_chars: int = None, 
    max_tokens: int = None):

    for text in texts:
        if not isinstance(text, str):
            print(type(text))
            exit()

    tokenizer = AutoTokenizer.from_pretrained(
        "dumitrescustefan/bert-base-romanian-uncased-v1"
    )
    text_tokens: List[Union[List[int], None]] = []
    max_encoding_len = 0
    for i in trange(len(texts), desc='(Preprocessing) Text with Romanian BERT'):
        text = texts[i]

        if not max_chars is None and len(text) > max_chars:
            text_tokens.append(None)
        else:
            encoded: List[int] = tokenizer.encode(text, add_special_tokens=True)

            if not max_tokens is None and len(encoded) > max_tokens:
                text_tokens.append(None)
            else:
                text_tokens.append(encoded)
                max_encoding_len = max(max_encoding_len, len(encoded))

    return text_tokens, max_encoding_len

def pad_bert_tokens(text_tokens: List[Union[List[int], None]] , 
    max_encoding_len: int):

    for i, encoded in enumerate(text_tokens):
        if encoded is None:
            continue

        if len(encoded) < max_encoding_len:
            encoded.extend([0] * (max_encoding_len - len(encoded)))

        encoded = torch.tensor(encoded)
        text_tokens[i] = encoded

    return text_tokens

class MultimodalDataset(Dataset):

    # https://github.com/pytorch/pytorch/issues/1137#issuecomment-299520530
    def __init__(self, df: pd.DataFrame, save_dir: str, 
        img_transform: transforms.Compose, metadata: MetaData, force=False):

        self.save_dir = save_dir
        self.metadata = metadata

        present_domains: List[str] = df['topic'].unique().tolist()
        present_domains.sort()
        self.domain_remapping: Dict[int, int] = {
            metadata.domain_mapping[domain_name]: index \
                for index, domain_name in enumerate(present_domains)
        }

        # domain remapping makes sure the domains that are present in this subset 
        # are indexed starting from 0 without skips; for example, if domain 3 and 
        # 5 are present then they'll be mapped to 0 and 1 respectively; this is 
        # done to correctly compute the domain loss
        df['domain'] = df['domain'].apply(
            lambda domain_index: self.domain_remapping[domain_index]
        )

        self.img_transform = img_transform

        self.text_tokens = self._load_text_tokens(df, force=force)

        self.image_paths: List[str] = df['image_path'].tolist()

        self.domains = df['domain'].tolist()

        self.labels = df['label'].tolist()

        # remove entries that were too large
        i = 0
        while i < len(self.text_tokens):
            if self.text_tokens[i] is None:
                del self.text_tokens[i]
                del self.image_paths[i]
                del self.domains[i]
                del self.labels[i]
            else:
                i += 1

    def _load_text_tokens(self, df: pd.DataFrame, force=False):
        file_path = os.path.join(self.save_dir, 'bert_tokens.pt')

        if not force and os.path.exists(file_path):
            return torch.load(file_path)

        text_tokens, max_encoded_len = bert_tokenize(
            df['text'].tolist(),
            self.metadata.max_chars,
            self.metadata.max_tokens
        )
        text_tokens = pad_bert_tokens(text_tokens, max_encoded_len)

        torch.save(text_tokens, file_path)

        return text_tokens

    def _get_image(self, idx: int):
        image_path = self.image_paths[idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            image = Image.open(image_path).convert('RGB')
            image = self.img_transform(image)

        return image

    def __len__(self):
        assert len(self.text_tokens) == len(self.image_paths)
        return len(self.text_tokens)

    def __getitem__(self, idx: int):
        return self.text_tokens[idx], self._get_image(idx), \
            torch.tensor(self.labels[idx]), torch.tensor(self.domains[idx])

    def print_stats(self):
        by_label = {'name': 'Sorted by label'}
        by_domain = {'name': 'Sorted by domain'}
        label_only = {'name': 'Sorted by label only'}
        domain_only = {'name': 'Sorted by domain only'}

        reverse_label_mapping = self.metadata.reverse_label_mapping()
        reverse_domain_mapping = self.metadata.reverse_domain_mapping()
        reverse_domain_remapping = {
            self.domain_remapping[index]: index for index in self.domain_remapping
        }

        for i in range(len(self.labels)):
            label = reverse_label_mapping[self.labels[i]]
            domain_global_index = reverse_domain_remapping[self.domains[i]]
            domain = reverse_domain_mapping[domain_global_index]

            domain_dict = by_label.get(label, {})
            domain_dict[domain] = 1 + domain_dict.get(domain, 0)
            by_label[label] = domain_dict 

            label_dict = by_domain.get(domain, {})
            label_dict[label] = 1 + label_dict.get(label, 0)
            by_domain[domain] = label_dict

            label_only[label] = 1 + label_only.get(label, 0)
            domain_only[domain] = 1 + domain_only.get(domain, 0)

        print(json.dumps(by_label, indent=4))
        print(json.dumps(by_domain, indent=4))
        print(json.dumps(label_only, indent=4))
        print(json.dumps(domain_only, indent=4))

class MappingDataset(Dataset):
    def __init__(self, dataset: Dataset, mapping: List[int]):
        self.dataset = dataset
        self.mapping = mapping

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx: int):
        return self.dataset[self.mapping[idx]]

def train_validate_split(dataset: Dataset, train_ratio = 0.9, seed = 42):
    random_permutation = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(random_permutation)

    split_index = int(len(dataset) * train_ratio)

    train_subset = MappingDataset(dataset, random_permutation[:split_index])
    valid_subset = MappingDataset(dataset, random_permutation[split_index:])

    return train_subset, valid_subset

def create_subset(
    subset: Union[str, pd.DataFrame],
    images_root_path: str,
    subset_save_dir: str,
    name: str,
    metadata: MetaData,
    force=False
):
    if isinstance(subset, pd.DataFrame):
        df = subset
    else:
        df = pd.read_csv(subset, delimiter='\t')

    # domain mapping turns the domain name (string) into an index (positive integer)
    df['domain'] = df['topic'].apply(
        lambda domain_name: metadata.domain_mapping[domain_name]
    )

    df['label'] = df['sarcastic'].apply(
        lambda label: metadata.label_mapping[label]
    )

    df['image_path'] = df['image_path'].apply(
        lambda image_path: os.path.abspath(os.path.join(images_root_path, image_path))
    )

    subset_save_dir = os.path.join(subset_save_dir, name)
    if not os.path.exists(subset_save_dir):
        os.makedirs(subset_save_dir)

    img_transform = create_image_transform(metadata)

    return MultimodalDataset(df, subset_save_dir, img_transform, metadata, force=force)

def load_subset(dataset: MultimodalDataset, batch_size: int, shuffle: bool,
    num_workers = 8):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
