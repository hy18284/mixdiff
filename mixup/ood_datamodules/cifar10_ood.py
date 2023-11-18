import random
from typing import (
    Callable,
    Optional,
) 
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage

import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose, 
    Resize, 
    CenterCrop, 
    ToTensor, 
    Normalize, 
)
from PIL import Image

from dataloaders.ZO_Clip_loaders import cifar10_single_isolated_class_loader
from .base_ood_datamodule import BaseOODDataModule


class CIFAR10Wrapper(CIFAR10):
    clip_transform = Compose([
        # ToPILImage(),
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
    ])
    # def __init___(self, clip_transform: Callable=None, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     if clip_transform is not None:
    #         self.clip_transform = clip_transform

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return self.clip_transform(image), label
            


class CIFAR10OODDataset(BaseOODDataModule):
    def __init__(
        self, 
        transform: Optional[Callable] = None,
        train_transform: Optional[Callable] = None,
        drop_last: bool = False,
    ):
        self.drop_last = drop_last
        self.splits = [
            ['airplane', 'automobile', 'bird', 'deer', 'dog', 'truck', 'cat', 'frog', 'horse', 'ship'],
            ['airplane', 'cat', 'dog', 'horse', 'ship', 'truck', 'automobile', 'bird', 'deer', 'frog'],
            ['airplane', 'automobile', 'dog', 'frog', 'horse', 'ship', 'bird', 'cat', 'deer', 'truck'],
            ['cat', 'deer', 'dog', 'horse', 'ship', 'truck', 'airplane', 'automobile', 'bird', 'frog'],
            ['airplane', 'automobile', 'bird', 'cat', 'horse', 'ship', 'deer', 'dog', 'frog', 'truck'],
        ]
        self.num_known = 6

        # train_transform = Compose([
        #     ToPILImage(),
        #     Resize(224, interpolation=Image.BICUBIC),
        #     CenterCrop(224),
        #     ToTensor(),
        #     # Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        # ])

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
        self.cifar10 = CIFAR10(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform,
        )
        self.cifar10_loaders_train = cifar10_single_isolated_class_loader(
            train=True,
            transform=transform,
        )
        # if transform is not None:
        #     for loader in self.cifar10_loaders_train.values():
        #         loader.dataset.transform = transform

        loader = DataLoader(
            self.cifar10, 
            batch_size=batch_size, 
            num_workers=2, 
            shuffle=shuffle,
            drop_last=self.drop_last,
        )
        for i in range(len(self.splits)):
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

            seen_class_idx = self.convert_names_to_idx(seen_class_names)

            if self.ref_mode in ('oracle', 'in_batch'):
                ref_images = None
            elif self.ref_mode == 'rand_id':
                ref_images = torch.cat(ref_images, dim=0)
                ref_images = random.Random(seed).choices(ref_images, k=n_ref_samples)
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
            loader = self.cifar10_loaders_train[seen_class_name]
            images = random.Random(seed).choices(loader.dataset, k=n_samples_per_class)
            images = torch.stack(images)
            given_images.append(images)
        
        # (NC, M, C, W, H)
        given_images = torch.stack(given_images)
        return given_images
    
    def convert_names_to_idx(self, seen_class_names: list[str]):
        seen_idx = [self.cifar10.class_to_idx[seen_label] for seen_label in seen_class_names]
        seen_idx = torch.tensor(seen_idx)
        return seen_idx
    
    def get_seen_class_names(self, i: int):
        split = self.splits[i]
        seen_class_names = split[:self.num_known]
        return seen_class_names
    
    def construct_loader(self, batch_size: int, shuffle: bool = True):
        loader = DataLoader(
            self.cifar10, 
            batch_size=batch_size, 
            num_workers=2, 
            shuffle=shuffle,
            drop_last=self.drop_last,
        )
        return loader
    
    def __str__(self) -> str:
        return 'cifar10'