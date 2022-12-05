import random
from pathlib import Path
from collections import defaultdict
import itertools
import copy

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
        self.transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])

        dataset = Caltech101(
            root='./data',
            target_type='category',
            download=True,
        )

        self.class2idx = make_class2idx_map(dataset) 
        class2data = split_by_class(dataset)
        self.train_dict, val_dict = split_train_val(
            class2data, 
            ratio=VAL_RATIO, 
            seed=SEED,
        )
        self.val = SimpleCaltech101(copy.deepcopy(val_dict), self.transform)

        self.seen_classes = []
        for seed in range(NUM_SPLITS):
            seen_classes = random.Random(seed).choices(
                list(self.class2idx.keys()), 
                k=20,
            )
            self.seen_classes.append(seen_classes)

        self.train_per_class = {
            name: SimpleCaltech101(
                {name: items},
                self.transform,
            )
            for name, items in self.train_dict.items() 
        }
        
    def get_splits(self, n_samples_per_class: int, seed: int):
        for i in range(len(self.seen_classes)):
            seen_class_names = self.seen_classes[i]
            seen_class_idx = [self.class2idx[name] for name in seen_class_names]
            seen_class_idx = torch.tensor(seen_class_idx)
            given_images = self.sample_given_images(
                seen_class_names=seen_class_names,
                n_samples_per_class=n_samples_per_class,
                seed=seed,
            )
            seen_class_names = [name.replace('_', ' ') for name in seen_class_names]

            yield seen_class_names, seen_class_idx, given_images

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
    
    def construct_loader(self, batch_size: int):
        loader = DataLoader(self.val, batch_size=batch_size, num_workers=2, shuffle=True)
        return loader
    
    def __str__(self) -> str:
        return 'caltech101'
    