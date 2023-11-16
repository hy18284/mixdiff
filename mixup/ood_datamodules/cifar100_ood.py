from typing import (
    Optional,
    Callable,
)
import random
import numpy as np

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
    def __init__(self, max_split: Optional[int] = None, with_replacement: bool = True):
        self.with_replacement = with_replacement
        self.cifar100_loaders_train = cifar100_single_isolated_class_loader(train=True)
        self.splits = [
            list(range(20)), 
            list(range(20, 40)), 
            list(range(40, 60)), 
            list(range(60, 80)), 
            list(range(80, 100))
        ]
        self.max_split = max_split

    def get_splits(
        self, 
        n_samples_per_class: int, 
        seed: int, 
        n_ref_samples: int,
        batch_size: int,
        shuffle: bool = True,
        transform: Optional[Callable] = None,
        n_few_shot_samples: Optional[int] = None,
    ):
        self.cifar100 = CIFAR100(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform,
        )
        self.idx2class = {v:k for k,v in self.cifar100.class_to_idx.items()}
        loader = DataLoader(
            self.cifar100, 
            batch_size=batch_size, 
            num_workers=2, 
            shuffle=shuffle,
        )

        for i in range(len(self.splits)):

            if self.max_split is not None and i >= self.max_split:
                break

            seen_class_names = self.get_seen_class_names(i)

            id_imgs_per_class = self.sample_given_images(
                seen_class_names, 
                n_samples_per_class + n_ref_samples,
                seed,
            )

            ref_images, given_images = [], []
            for id_images in id_imgs_per_class:
                ref_images.append(id_images[n_samples_per_class:])
                given_images.append(id_images[:n_samples_per_class])
            given_images = torch.stack(given_images)

            seen_class_idx = torch.tensor(self.splits[i])

            if self.ref_mode in ('oracle', 'in_batch'):
                ref_images = None
            elif self.ref_mode == 'rand_id':
                ref_images = torch.cat(ref_images, dim=0)
                if self.with_replacement:
                    ref_images = random.Random(seed).choices(ref_images, k=n_ref_samples)
                else:
                    ref_images_idx = np.random.default_rng(seed).choice(
                        len(ref_images),
                        size=n_ref_samples,
                        replace=False,
                    )
                    ref_images = [ref_images[idx] for idx in ref_images_idx]

                ref_images = torch.stack(ref_images)
            else:
                raise ValueError()

            yield seen_class_names, seen_class_idx, given_images, ref_images, None, loader, None

    def sample_given_images(
        self, 
        seen_class_names: list[str],
        n_samples_per_class: int,
        seed: int,
    ):
        given_images = []
        for seen_class_name in seen_class_names:
            loader = self.cifar100_loaders_train[seen_class_name]
            if self.with_replacement:
                images = random.Random(seed).choices(loader.dataset, k=n_samples_per_class)
            else:
                images_idx = np.random.default_rng(seed).choice(
                    len(loader.dataset),
                    size=n_samples_per_class,
                    replace=False,
                )
                images = [loader.dataset[idx] for idx in images_idx]
            images = torch.stack(images)
            given_images.append(images)
        
        # (NC, M, C, W, H)
        given_images = torch.stack(given_images)
        return given_images
    
    def get_seen_class_names(self, i: int):
        split = self.splits[i]
        seen_class_names = [self.idx2class[idx] for idx in split]
        return seen_class_names
    
    def construct_loader(self, batch_size: int, shuffle: bool = True):
        loader = DataLoader(
            self.cifar100, 
            batch_size=batch_size, 
            num_workers=2, 
            shuffle=shuffle,
        )
        return loader
    
    def __str__(self) -> str:
        return 'cifar100'