from typing import (
    Optional,
    Callable,
)
import random
from pathlib import Path
from collections import defaultdict
import itertools
import copy
import math

import torch
from torchvision.datasets import Caltech101
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from torchvision.transforms import (
    Compose, 
    Resize, 
    CenterCrop, 
    ToTensor, 
    Normalize, 
)
from PIL import Image

from .base_ood_datamodule import BaseOODDataModule


SEED = 0
VAL_RATIO = 0.66
NUM_SPLITS = 3


def make_class2idx_map(dataset):
    class2idx = {}
    for img, idx in dataset:
        class_name = Path(img.filename).parent.name
        class2idx[class_name] = idx
    return class2idx


def split_by_class(dataset):
    class2data = defaultdict(list)
    for img, idx in dataset:
        class_name = Path(img.filename).parent.name
        class2data[class_name].append((img, idx))
    return class2data


def split_train_val(class2data, ratio, seed):
    train = {}
    val = {}
    for name, samples in class2data.items():
        random.Random(seed).shuffle(samples)
        num_val = int(len(samples) * ratio)
        val[name] = samples[:num_val]
        train[name] = samples[num_val:]
    return train, val


class SimpleCaltech101(Dataset):
    def __init__(self, samples_dict, transform):
        self.transform = transform

        data = samples_dict.values()
        data = itertools.chain.from_iterable(data)
        self.data = list(data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image), label
    
    def __len__(self):
        return len(self.data)


class Caltech101OODDataset(BaseOODDataModule):
    def __init__(self, ):
        dataset = Caltech101(
            root='./data',
            target_type='category',
            download=True,
        )

        self.class2idx = make_class2idx_map(dataset) 
        class2data = split_by_class(dataset)
        self.train_dict, self.val_dict = split_train_val(
            class2data, 
            ratio=VAL_RATIO, 
            seed=SEED,
        )

        self.seen_classes = []
        for seed in range(NUM_SPLITS):
            seen_classes = random.Random(seed).choices(
                list(self.class2idx.keys()), 
                k=20,
            )
            self.seen_classes.append(seen_classes)

    def get_splits(
        self, 
        n_samples_per_class: int, 
        seed: int, 
        n_ref_samples: int,
        batch_size: int,
        shuffle: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.val = SimpleCaltech101(copy.deepcopy(self.val_dict), transform)
        self.train_per_class = {
            name: SimpleCaltech101(
                {name: items},
                transform,
            )
            for name, items in self.train_dict.items() 
        }
        loader = DataLoader(
            self.val, 
            batch_size=batch_size, 
            num_workers=2, 
            shuffle=shuffle,
        )

        for i in range(len(self.seen_classes)):
            seen_class_names = self.seen_classes[i]
            seen_class_idx = [self.class2idx[name] for name in seen_class_names]
            seen_class_idx = torch.tensor(seen_class_idx)

            id_imgs_per_class = self.sample_given_images(
                seen_class_names=seen_class_names,
                n_samples_per_class=n_samples_per_class + math.ceil(n_ref_samples / len(seen_class_names)),
                seed=seed,
            )

            ref_images, given_images = [], []
            for id_images in id_imgs_per_class:
                ref_images.append(id_images[n_samples_per_class:])
                given_images.append(id_images[:n_samples_per_class])
            given_images = torch.stack(given_images)

            seen_class_names = [name.replace('_', ' ') for name in seen_class_names]

            if self.ref_mode in ('oracle', 'in_batch'):
                ref_images = None
            elif self.ref_mode == 'rand_id':
                ref_images = torch.cat(ref_images, dim=0)
                ref_images = random.Random(seed).choices(ref_images, k=n_ref_samples)
                ref_images = torch.stack(ref_images)
            else:
                raise ValueError()

            yield seen_class_names, seen_class_idx, given_images, ref_images, None, loader

    def sample_given_images(
        self, 
        seen_class_names: list[str],
        n_samples_per_class: int,
        seed: int,
    ):
        given_images = []
        for seen_class_name in seen_class_names:
            dataset = self.train_per_class[seen_class_name]
            pairs = random.Random(seed).choices(dataset, k=n_samples_per_class)
            images = [image for image, _ in pairs]
            images = torch.stack(images)
            given_images.append(images)
        
        # (NC, M, C, W, H)
        given_images = torch.stack(given_images)
        return given_images
    
    def construct_loader(self, batch_size: int, shuffle: bool = True):
        loader = DataLoader(
            self.val, 
            batch_size=batch_size, 
            num_workers=2, 
            shuffle=shuffle,
        )
        return loader
    
    def __str__(self) -> str:
        return 'caltech101'
    