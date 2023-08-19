import random
from typing import (
    Callable,
    Optional,
) 
from collections import defaultdict

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import (
    DataLoader,
    random_split,
    Dataset,
    ConcatDataset,
)
from torchvision import transforms
from torchvision.datasets import ImageNet
from PIL import Image
from sklearn.model_selection import train_test_split

from .base_ood_datamodule import BaseOODDataModule


class CrossDatasetOODDataset(BaseOODDataModule):
    def __init__(
        self, 
        id_dataset_dir: str,
        ood_dataset_dir: str,
        name: Optional[str] = None
    ):
        self.name = name

        self.id_dataset = ImageFolder(
            id_dataset_dir,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        )
        self.seen_idx = list(set(self.id_dataset.class_to_idx.values()))
        self.seen_idx.sort()
        self.seen_idx = torch.tensor(self.seen_idx)
        self.seen_idx = torch.arange(1000)
        print(self.seen_idx.tolist())
        print(self.id_dataset.class_to_idx)
        print(len(self.seen_idx))
        print(len(self.id_dataset))
        

        self.ood_dataset = ImageFolder(
            ood_dataset_dir,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            target_transform=lambda x: x + len(self.seen_idx)
        )
        
    def get_splits(
        self, 
        n_samples_per_class: int, 
        seed: int, 
        n_ref_samples: int,
        batch_size: int,
        shuffle: bool = True,
    ):

        self.id_datasets = random_split(
            self.id_dataset, 
            [0.5, 0.5], 
            generator=torch.Generator().manual_seed(seed),
        )

        self.ood_datasets = random_split(
            self.ood_dataset, 
            [0.5, 0.5], 
            generator=torch.Generator().manual_seed(seed),
        )

        for i, (id_dataset, ood_dataset) in enumerate(zip(self.id_datasets, self.ood_datasets)):
            given_images, ref_images = self._sample_given_images(
                dataset=self.id_datasets[(i + 1) % len(self.id_datasets)],
                n_samples_per_class=n_samples_per_class,
                n_ref_samples=n_ref_samples,
                seed=seed,
            )

            if self.ref_mode in ('oracle', 'in_batch'):
                ref_images = None
            elif self.ref_mode == 'rand_id':
                ref_images = torch.cat(ref_images, dim=0)
                ref_images = random.Random(seed).choices(ref_images, k=n_ref_samples)
                ref_images = torch.stack(ref_images)
            else:
                raise ValueError()

            loader = DataLoader(
                ConcatDataset([id_dataset, ood_dataset]),
                batch_size=batch_size, 
                num_workers=2, 
                shuffle=shuffle
            )
            yield None, self.seen_idx, given_images, ref_images, None, loader

    def _sample_given_images(
        self, 
        dataset: Dataset,
        n_samples_per_class: int,
        n_ref_samples: int,
        seed: int,
    ):
        label_2_images = defaultdict(list)
        filled_labels = set()
        for image, label in dataset:
            # print(len(label_2_images))
            # print(filled_labels)
            label_2_images[label].append(image)
            if len(label_2_images[label]) >= n_samples_per_class + n_ref_samples:
                filled_labels.add(label)
            if len(filled_labels) >= len(self.seen_idx):
                break

        given_images_list = []
        ref_images = []
        for label in range(len(self.seen_idx)):
            images = label_2_images[label]
            random.Random(seed).shuffle(images)
            print(len(images), n_samples_per_class, n_ref_samples)
            assert len(images) >= (n_samples_per_class + n_ref_samples)
            given_images = images[:n_samples_per_class]
            given_images = torch.stack(given_images)
            given_images_list.append(given_images)

            if n_ref_samples > 0:
                ref_images.extend(images[-n_ref_samples:])
        
        # (NC, M, C, W, H)
        given_images = torch.stack(given_images_list)

        if n_ref_samples > 0:
            # (NC * P, C, W, H)
            ref_images = torch.stack(ref_images)
        else:
            ref_images = None

        return given_images, ref_images
    
    # def post_transform(self, images):
    #     orig_size = images.size()
    #     C, H, W = orig_size[-3:]
    #     images = images.view(-1, C, H, W)
    #     images = self._post_transform(images)
    #     images = images.view(orig_size)
    #     return images
    
    @property
    def flatten(self):
        return True
    
    def __str__(self) -> str:
        return self.name