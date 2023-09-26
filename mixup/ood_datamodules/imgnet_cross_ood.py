import random
from typing import (
    Callable,
    Optional,
) 
from collections import defaultdict
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import (
    DataLoader,
    random_split,
    Dataset,
    ConcatDataset,
)
from PIL import Image

from .base_ood_datamodule import BaseOODDataModule


class SplitOODDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        seed: int, 
        split_idx: int,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.dataset = ImageFolder(data_dir)
        
        class2paths = defaultdict(list)
        for path, label in tqdm(self.dataset.imgs, desc='Collecting images for oracle split...'):
            class2paths[label].append((path, label))
        
        rng = np.random.default_rng(seed)
        for label in class2paths.keys():
            rng.shuffle(class2paths[label])
            idx = len(class2paths[label]) // 2
            if split_idx == 0:
                class2paths[label] = class2paths[label][:idx]
            elif split_idx == 1:
                class2paths[label] = class2paths[label][idx:]
            else:
                raise ValueError(f'Invalid split_idx {split_idx}')

        self.items = itertools.chain.from_iterable(class2paths.values())
        self.items = list(self.items)
        rng.shuffle(self.items)

    def __getitem__(self, idx):
        image, label = self.items[idx]
        return (
            self.transform(Image.open(image).convert('RGB')),
            label,
        )
    
    def __len__(self):
        return len(self.items)
    
    def label_to_class_id(self):
        label_to_class_id = {}
        for path, label in tqdm(self.dataset.imgs, desc='Collecting images for oracle split...'):
            if label not in label_to_class_id:
                label_to_class_id[label] = Path(path).parents[0].name
            if len(label_to_class_id) >= 1000:
                break
        return label_to_class_id


class ImageNetCrossOODDataset(BaseOODDataModule):
    def __init__(
        self, 
        ood_dataset_dir: str,
        id_dataset_dir: str = 'data/imagenet/val',
        name: Optional[str] = None,
        class_name_path: str = 'data/imagenet/imagenet_class_index.json'
    ):
        self.id_dataset_dir = id_dataset_dir
        self.ood_dataset_dir = ood_dataset_dir
        self.class_name_path = class_name_path
        self.name = name

        # # TODO: May not match with class label indices.
        # with open(class_name_path, 'rb') as f:
        #     self.class_names = list(np.load(f))
        
        self.seen_idx = torch.arange(1000)

        
    def get_splits(
        self, 
        n_samples_per_class: int, 
        seed: int, 
        n_ref_samples: int,
        batch_size: int,
        shuffle: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.id_datasets = [
            SplitOODDataset(
                data_dir=self.id_dataset_dir,
                seed=seed,
                split_idx=split_idx,
                transform=transform,
            )
            for split_idx in range(2)
        ]
        label_to_class_id = self.id_datasets[0].label_to_class_id()
        with open(self.class_name_path) as f:
            label_to_name = json.load(f)

        class_id_to_name = {}
        for class_id, class_name in label_to_name.values():
            class_id_to_name[class_id] = class_name

        self.class_names = []
        for label in range(1000):
            class_name = class_id_to_name[label_to_class_id[label]]
            self.class_names.append(class_name.replace('_', ' '))

        self.ood_dataset = ImageFolder(
            self.ood_dataset_dir,
            transform=transform,
            target_transform=lambda x: x + len(self.seen_idx)
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
                n_ref_samples=int(np.ceil(n_ref_samples / len(self.seen_idx))),
                seed=seed,
            )

            if self.ref_mode in ('oracle', 'in_batch'):
                ref_images = None
            elif self.ref_mode == 'rand_id':
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
            yield self.class_names, self.seen_idx, given_images, ref_images, None, loader

    def _sample_given_images(
        self, 
        dataset: Dataset,
        n_samples_per_class: int,
        n_ref_samples: int,
        seed: int,
    ):
        label_2_images = defaultdict(list)
        filled_labels = set()
        for image, label in tqdm(dataset, desc='Selecting oracle samples...'):
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
    
    @property
    def flatten(self):
        return True
    
    def __str__(self) -> str:
        return self.name