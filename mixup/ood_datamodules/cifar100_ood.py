import random

import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose, 
    Resize, 
    CenterCrop, 
    ToTensor, 
    Normalize, 
)
from PIL import Image

from dataloaders.ZO_Clip_loaders import cifar100_single_isolated_class_loader
from .base_ood_datamodule import BaseOODDataModule


class CIFAR100Wrapper(CIFAR100):
    clip_transform = Compose([
        # ToPILImage(),
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
    ])

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return self.clip_transform(image), label


class CIFAR100OODDataset(BaseOODDataModule):
    def __init__(self, ):
        self.cifar100_loaders_train = cifar100_single_isolated_class_loader(train=True)
        self.cifar100 = CIFAR100Wrapper(root='./data', train=False, download=True)
        self.idx2class = {v:k for k,v in self.cifar100.class_to_idx.items()}
        self.splits = [
            list(range(20)), 
            list(range(20, 40)), 
            list(range(40, 60)), 
            list(range(60, 80)), 
            list(range(80, 100))
        ]

    def get_splits(self, n_samples_per_class: int, seed: int):
        for i in range(len(self.splits)):
            seen_class_names = self.get_seen_class_names(i)
            given_images = self.sample_given_images(
                seen_class_names, 
                n_samples_per_class,
                seed,
            )
            seen_class_idx = torch.tensor(self.splits[i])

            yield seen_class_names, seen_class_idx, given_images

    def sample_given_images(
        self, 
        seen_class_names: list[str],
        n_samples_per_class: int,
        seed: int,
    ):
        given_images = []
        for seen_class_name in seen_class_names:
            loader = self.cifar100_loaders_train[seen_class_name]
            images = random.Random(seed).choices(loader.dataset, k=n_samples_per_class)
            images = torch.stack(images)
            given_images.append(images)
        
        # (NC, M, C, W, H)
        given_images = torch.stack(given_images)
        return given_images
    
    def get_seen_class_names(self, i: int):
        split = self.splits[i]
        seen_class_names = [self.idx2class[idx] for idx in split]
        return seen_class_names
    
    def construct_loader(self, batch_size: int):
        loader = DataLoader(self.cifar100, batch_size=batch_size, num_workers=2, shuffle=True)
        return loader
    
    def __str__(self) -> str:
        return 'cifar100'