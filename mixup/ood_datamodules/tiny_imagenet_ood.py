import random
import glob
import os

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from torchvision.transforms import (
    Compose, 
    Resize, 
    CenterCrop, 
    ToTensor, 
    Normalize, 
)
from PIL import Image

from dataloaders.ZO_Clip_loaders import tinyimage_single_isolated_class_loader
from .base_ood_datamodule import BaseOODDataModule


class TinyImageNetOODDataset(BaseOODDataModule):
    def __init__(self, ):
        self.tiny_imagenet = ImageFolder(
            root='./data/tiny-imagenet-200/val',
            transform=Compose([
                Resize(224, interpolation=Image.BICUBIC),
                CenterCrop(224),
                ToTensor(),
                Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
            ]),
        )
        self.class2dir = {}
        with open('dataloaders/tinyimagenet_labels_to_ids.txt') as f:
            for line in f:
                class_name, dir_name = line.strip().split()
                self.class2dir[class_name] = dir_name

        self.class_names, self.loaders_train = tinyimage_single_isolated_class_loader(train=True)
        print(self.class_names)

    def get_splits(
        self, 
        n_samples_per_class: int, 
        seed: int, 
        n_ref_samples: int,
        batch_size: int,
        shuffle: bool = True,
    ):
        loader = DataLoader(
            self.tiny_imagenet, 
            batch_size=batch_size, 
            num_workers=2, 
            shuffle=shuffle,
        )
        for i in range(len(self.class_names)):
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
            seen_class_idx = self.get_seen_class_idx(seen_class_names)

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
            loader = self.loaders_train[seen_class_name]
            images = random.Random(seed).choices(loader.dataset, k=n_samples_per_class)
            images = torch.stack(images)
            given_images.append(images)
        
        # (NC, M, C, W, H)
        given_images = torch.stack(given_images)
        return given_images
    
    def get_seen_class_names(self, i: int):
        seen_class_names = self.class_names[i][:20]
        return seen_class_names
    
    def get_seen_class_idx(self, seen_class_names):
        seen_class_idx = [
            self.tiny_imagenet.class_to_idx[self.class2dir[name]]
            for name in seen_class_names
        ]
        return torch.tensor(seen_class_idx)
    
    def construct_loader(self, batch_size: int, shuffle: bool = True):
        loader = DataLoader(
            self.tiny_imagenet, 
            batch_size=batch_size, 
            num_workers=2, 
            shuffle=shuffle,
        )
        return loader
    
    def __str__(self) -> str:
        return 'tiny_imagenet'