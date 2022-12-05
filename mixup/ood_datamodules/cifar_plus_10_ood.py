import random

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
        self.transform = Compose([
            ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])
    
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
    def __init__(self, class_idx):
        self.transform = Compose([
            ToPILImage(),
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])
        cifar10 = CIFAR10(root='./data', train=True, download=True)
        inds = [
            i for i in range(len(cifar10.targets)) 
            if cifar10.targets[i] == class_idx
        ]
        self.data = cifar10.data[inds]
        self.targets = np.array(cifar10.targets)[inds].tolist()

    def __getitem__(self, idx):
        return self.transform(self.data[idx])
    
    def __len__(self):
        return len(self.data)


class CIFARPlus10OODDataset(BaseOODDataModule):
    def __init__(self, ):
        self.seen_class_names = ['airplane', 'automobile', 'ship', 'truck']
        in_loader, out_loaders = cifarplus_loader()
        self.in_dataset = in_loader.dataset
        self.out_datasets = [
            loader.dataset for key, loader in out_loaders.items() if 'plus10' in key
        ]
        self.seen_class_idx = torch.tensor([0, 1, 8, 9])
        self.partial_cifar10_train = {
            seen_class_idx: PartialCIFAR10(seen_class_idx) 
            for seen_class_idx in self.seen_class_idx.tolist()
        }
        self.cur_loader_idx = 0

    def get_splits(self, n_samples_per_class: int, seed: int):
        for _ in range(len(self.out_datasets)):
            seen_class_names = self.seen_class_names
            given_images = self.sample_given_images(
                self.seen_class_idx.tolist(), 
                n_samples_per_class,
                seed,
            )
            seen_class_idx = self.seen_class_idx

            yield seen_class_names, seen_class_idx, given_images

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
    
    def construct_loader(self, batch_size: int):
        cifar_plus_10 = CIFARPlus(
            self.in_dataset,
            self.out_datasets[self.cur_loader_idx],
        )
        loader = DataLoader(
            cifar_plus_10, 
            batch_size=batch_size, 
            num_workers=2, 
            shuffle=True
        )
        self.cur_loader_idx += 1
        return loader
    
    def __str__(self) -> str:
        return 'cifar+10'