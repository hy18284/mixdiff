import random
from typing import (
    Optional,
    Callable,
)

import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import (
    DataLoader,
    Dataset
)
from torchvision.transforms import (
    Compose, 
    Resize, 
    CenterCrop, 
    ToTensor, 
    Normalize, 
    ToPILImage,
)
from PIL import Image
import numpy as np

from dataloaders.ZO_Clip_loaders import cifarplus_loader
from .base_ood_datamodule import BaseOODDataModule


class CIFARPlus(Dataset):
    def __init__(self, in_dataset, out_dataset):
        super().__init__()
        self.in_dataset = in_dataset
        self.out_dataset = out_dataset
    
    def __len__(self):
        return len(self.in_dataset) + len(self.out_dataset)

    def __getitem__(self, idx):
        if idx < len(self.in_dataset):
            image, label = self.in_dataset[idx]
        else:
            idx = idx - len(self.in_dataset)
            image, label = self.out_dataset[idx]
            label += 100
        return image, label


class PartialCIFAR10:
    def __init__(self, class_idx, transform = lambda x: x):
        # self.transform = Compose([
        #     ToPILImage(),
        #     Resize(224, interpolation=Image.BICUBIC),
        #     CenterCrop(224),
        #     ToTensor(),
        #     Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        # ])
        self.to_pil = ToPILImage()
        self.transform = transform
        cifar10 = CIFAR10(root='./data', train=True, download=True)
        inds = [
            i for i in range(len(cifar10.targets)) 
            if cifar10.targets[i] == class_idx
        ]
        self.data = cifar10.data[inds]
        self.targets = np.array(cifar10.targets)[inds].tolist()

    def __getitem__(self, idx):
        return self.transform(self.to_pil(self.data[idx]))
    
    def __len__(self):
        return len(self.data)


class CIFARPlus10OODDataset(BaseOODDataModule):
    def __init__(
        self, 
        drop_last: bool = False,
    ):
        self.drop_last = drop_last
        self.seen_class_names = ['airplane', 'automobile', 'ship', 'truck']
        self.seen_class_idx = torch.tensor([0, 1, 8, 9])
        self.cur_loader_idx = 0

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
        in_loader, out_loaders = cifarplus_loader(transform)
        self.in_dataset = in_loader.dataset
        self.out_datasets = [
            loader.dataset for key, loader in out_loaders.items() if 'plus10' in key
        ]

        self.partial_cifar10_train = {
            seen_class_idx: PartialCIFAR10(seen_class_idx, transform) 
            for seen_class_idx in self.seen_class_idx.tolist()
        }

        for i, _ in enumerate(range(len(self.out_datasets))):
            cifar_plus_10 = CIFARPlus(
                self.in_dataset,
                self.out_datasets[i],
            )
            loader = DataLoader(
                cifar_plus_10, 
                batch_size=batch_size, 
                num_workers=2, 
                shuffle=shuffle,
            )

            seen_class_names = self.seen_class_names
            id_imgs_per_class = self.sample_given_images(
                self.seen_class_idx.tolist(), 
                n_samples_per_class + n_ref_samples,
                seed,
            )
            ref_images, given_images = [], []
            for id_images in id_imgs_per_class:
                ref_images.append(id_images[n_samples_per_class:])
                given_images.append(id_images[:n_samples_per_class])
            given_images = torch.stack(given_images)

            seen_class_idx = self.seen_class_idx

            if self.ref_mode in ('oracle', 'in_batch'):
                ref_images = None
            elif self.ref_mode == 'rand_id':
                ref_images = torch.cat(ref_images, dim=0)
                ref_images = random.Random(seed).choices(ref_images, k=n_ref_samples)
                ref_images = torch.stack(ref_images)
            elif self.ref_mode == 'single_class':
                rng = np.random.default_rng(seed)
                idx = rng.integers(0, len(ref_images))
                ref_images = ref_images[idx]
                indices = rng.choice(len(ref_images), n_ref_samples, replace=False)
                ref_images = [ref_images[idx] for idx in indices]
                ref_images = torch.stack(ref_images)
            else:
                raise ValueError()

            yield seen_class_names, seen_class_idx, given_images, ref_images, None, loader, None

    def sample_given_images(
        self, 
        seen_class_indices: list[int],
        n_samples_per_class: int,
        seed: int,
    ):
        given_images = []
        for seen_class_idx in seen_class_indices:
            images = random.Random(seed).choices(
                self.partial_cifar10_train[seen_class_idx], 
                k=n_samples_per_class,
            )
            images = torch.stack(images)
            given_images.append(images)
        
        # (NC, M, C, W, H)
        given_images = torch.stack(given_images)
        return given_images
    
    def convert_names_to_idx(self, seen_class_names: list[str]):
        seen_idx = [self.cifar10.class_to_idx[seen_label] for seen_label in seen_class_names]
        seen_idx = torch.tensor(seen_idx)
        return seen_idx
    
    def construct_loader(self, batch_size: int, shuffle: bool = True):
        cifar_plus_10 = CIFARPlus(
            self.in_dataset,
            self.out_datasets[self.cur_loader_idx],
        )
        loader = DataLoader(
            cifar_plus_10, 
            batch_size=batch_size, 
            num_workers=2, 
            shuffle=shuffle,
        )
        self.cur_loader_idx += 1
        return loader
    
    def __str__(self) -> str:
        return 'cifar+10'