from typing import (
    Optional,
    Callable,
)
import random

import torch
from torch.utils.data import (
    DataLoader,
)

from dataloaders.ZO_Clip_loaders import cifarplus_loader
from .base_ood_datamodule import BaseOODDataModule
from .cifar_plus_10_ood import (
    PartialCIFAR10,
    CIFARPlus,
)
    

class CIFARPlus50OODDataset(BaseOODDataModule):
    def __init__(self):
        self.seen_class_names = ['airplane', 'automobile', 'ship', 'truck']
        in_loader, out_loaders = cifarplus_loader()
        self.in_dataset = in_loader.dataset
        self.out_datasets = [out_loaders['plus50'].dataset]
        self.seen_class_idx = torch.tensor([0, 1, 8, 9])
        self.partial_cifar10_train = {
            seen_class_idx: PartialCIFAR10(seen_class_idx) 
            for seen_class_idx in self.seen_class_idx.tolist()
        }
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
        for i, _ in enumerate(range(len(self.out_datasets))):
            cifar_plus_100 = CIFARPlus(
                self.in_dataset,
                self.out_datasets[i],
            )
            loader = DataLoader(
                cifar_plus_100, 
                batch_size=batch_size, 
                num_workers=2, 
                shuffle=shuffle
            )

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

            if self.ref_mode in ('oracle', 'in_batch'):
                ref_images = None
            elif self.ref_mode == 'rand_id':
                ref_images = torch.cat(ref_images, dim=0)
                ref_images = random.Random(seed).choices(ref_images, k=n_ref_samples)
                ref_images = torch.stack(ref_images)
            else:
                raise ValueError()

            yield self.seen_class_names, self.seen_class_idx, given_images, ref_images, None, loader, None

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
        seen_idx = [
            self.cifar10.class_to_idx[seen_label] 
            for seen_label in seen_class_names
        ]
        seen_idx = torch.tensor(seen_idx)
        return seen_idx
    
    def construct_loader(self, batch_size: int, shuffle: bool = True):
        cifar_plus_100 = CIFARPlus(
            self.in_dataset,
            self.out_datasets[self.cur_loader_idx],
        )
        loader = DataLoader(
            cifar_plus_100, 
            batch_size=batch_size, 
            num_workers=2, 
            shuffle=shuffle
        )
        self.cur_loader_idx += 1
        return loader
    
    def __str__(self) -> str:
        return 'cifar+50'