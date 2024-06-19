import random
from typing import (
    Callable,
    Optional,
) 
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage, RandomResizedCrop
import collections

import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    Compose, 
    Resize, 
    CenterCrop, 
    ToTensor, 
    Normalize, 
)
from PIL import Image
import numpy as np
import csv
import tqdm
 
from dataloaders.ZO_Clip_loaders import cifar10_single_isolated_class_loader
from .base_ood_datamodule import BaseOODDataModule

DX2LABEL = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2, 
    'df': 3,
    'nv': 4,
    'vasc': 5,
    'mel': 6,
}

LABEL2NAME = {
    0: 'actinic keratoses',  
    1: 'basal cell carcinoma', 
    2: 'benign keratosis-like lesions',
    3: 'dermatofibroma',
    4: 'melanocytic nevi',
    5: 'pyogenic granulomas and hemorrhage',
    6: 'melanoma',
}

NAME2LABEL= {
    'actinic keratoses': 0,  
    'basal cell carcinoma': 1, 
    'benign keratosis-like lesions': 2,
    'dermatofibroma': 3,
    'melanocytic nevi': 4,
    'pyogenic granulomas and hemorrhage': 5,
    'melanoma': 6,
}

NAME2DESC= {
    'actinic keratoses': 'Actinic keratoses appear as rough, scaly patches on sun-exposed areas of the skin. They can vary in color from skin-colored to reddish-brown and may feel rough and dry.',  
    'basal cell carcinoma': 'Basal cell carcinoma typically presents as a pearly or waxy bump, often with visible blood vessels on the surface. It may also appear as a flat, flesh-colored or brown scar-like lesion.', 
    'benign keratosis-like lesions': 'Benign keratosis-like lesions, such as seborrheic keratosis, are raised, waxy growths with a stuck-on appearance. They can vary in color from light tan to black and often have a rough texture.',
    'dermatofibroma': 'Dermatofibromas are benign skin growths that appear as small, firm nodules or plaques. They are usually brownish in color and may have a dimpled or depressed center when pinched.',
    'melanocytic nevi': 'Melanocytic nevi are benign growths composed of melanocytes (pigment-producing cells). They can vary in appearance from flat to raised, and in color from tan to dark brown.',
    'pyogenic granulomas and hemorrhage': 'Pyogenic granulomas are red, often rapidly growing nodules that can bleed easily. They are not cancerous but can be mistaken for more serious conditions due to their appearance.',
    'melanoma': 'Melanoma is a type of skin cancer that can develop from existing moles or appear as a new dark spot on the skin. It is characterized by asymmetry, irregular borders, uneven coloration, and a diameter larger than a pencil eraser (although not always).',
}


class HAM10000(Dataset):
    def __init__(self, metadata_path, img_dir, transform) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.img_id_dx_pairs = []
        with open(metadata_path) as f:
            metadata = csv.reader(f, delimiter=',')
            for i, line in enumerate(metadata):
                if i == 0:
                    continue
                img_id, dx = line[1:3]
                self.img_id_dx_pairs.append((img_id, dx)) 

    def __getitem__(self, idx):
        img_id, dx = self.img_id_dx_pairs[idx]
        img = Image.open(f'{self.img_dir}/{img_id}.jpg')
        img = self.transform(img)
        label = DX2LABEL[dx]
        return img, label
    
    def __len__(self):
        return len(self.img_id_dx_pairs)


class HAM10000OODDataset(BaseOODDataModule):
    def __init__(
        self, 
        drop_last: bool = False,
        add_desc: bool = False,
    ):
        self.drop_last = drop_last
        self.add_desc = add_desc
        self.splits = [
            [
                'actinic keratoses',
                'melanocytic nevi',
                'pyogenic granulomas and hemorrhage',
                'basal cell carcinoma',
                'benign keratosis-like lesions',
                'dermatofibroma',
                'melanoma'
            ],
            [
                'melanoma',
                'melanocytic nevi',
                'dermatofibroma',
                'actinic keratoses',
                'pyogenic granulomas and hemorrhage',
                'benign keratosis-like lesions',
                'basal cell carcinoma'
            ],
            [
                'actinic keratoses',
                'basal cell carcinoma',
                'melanoma',
                'pyogenic granulomas and hemorrhage',
                'benign keratosis-like lesions',
                'dermatofibroma',
                'melanocytic nevi'
            ],
            [
                'melanocytic nevi',
                'pyogenic granulomas and hemorrhage',
                'benign keratosis-like lesions',
                'melanoma',
                'basal cell carcinoma',
                'dermatofibroma',
                'actinic keratoses'
            ],
            [
                'pyogenic granulomas and hemorrhage',
                'actinic keratoses',
                'dermatofibroma',
                'melanocytic nevi',
                'basal cell carcinoma',
                'benign keratosis-like lesions',
                'melanoma'
            ],
        ]

        self.num_known = 4


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
        self.dataset = HAM10000(
            metadata_path='data/ham10000/ISIC2018_Task3_Test_GroundTruth.csv',
            img_dir='data/ham10000/ISIC2018_Task3_Test_Images',
            transform=transform,
        )

        self.train = HAM10000(
            metadata_path='data/ham10000/HAM10000_metadata.csv',
            img_dir='data/ham10000/images_all',
            transform=transform,
        )

        self.name2imgs = collections.defaultdict(list)
        for i in tqdm.tqdm(range(len(self.train)), desc='Constructing Name Mapping...'):
            img, label = self.train[i]
            name = LABEL2NAME[label]
            self.name2imgs[name].append(img)

        loader = DataLoader(
            self.dataset, 
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
            elif self.ref_mode == 'single_class':
                rng = np.random.default_rng(seed)
                idx = rng.integers(0, len(ref_images))
                ref_images = ref_images[idx]
                indices = rng.choice(len(ref_images), n_ref_samples, replace=False)
                ref_images = [ref_images[idx] for idx in indices]
                ref_images = torch.stack(ref_images)
            elif self.ref_mode == 'single_sample':
                ref_images = torch.cat(ref_images)
                rng = np.random.default_rng(seed)
                idx = rng.integers(0, len(ref_images))
                ref_image = ref_images[idx]
                crop = RandomResizedCrop(ref_image.size()[1:])
                ref_images = [crop(ref_image) for _ in range(n_ref_samples)]
                ref_images = torch.stack(ref_images)
            else:
                raise ValueError()

            if self.add_desc:
                seen_class_names = [f'{name}. {NAME2DESC[name]}' for name in seen_class_names]
            yield seen_class_names, seen_class_idx, given_images, ref_images, None, loader, None

    def sample_given_images(
        self, 
        seen_class_names: list[str],
        n_samples_per_class: int,
        seed: int,
    ):
        given_images = []
        for seen_class_name in seen_class_names:
            images = self.name2imgs[seen_class_name]
            images = random.Random(seed).choices(images, k=n_samples_per_class)
            images = torch.stack(images)
            given_images.append(images)
        
        # (NC, M, C, W, H)
        given_images = torch.stack(given_images)
        return given_images
    
    def convert_names_to_idx(self, seen_class_names: list[str]):
        seen_idx = [NAME2LABEL[seen_label] for seen_label in seen_class_names]
        seen_idx = torch.tensor(seen_idx)
        return seen_idx
    
    def get_seen_class_names(self, i: int):
        split = self.splits[i]
        seen_class_names = split[:self.num_known]
        return seen_class_names
    
    def construct_loader(self, batch_size: int, shuffle: bool = True):
        loader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            num_workers=2, 
            shuffle=shuffle,
            drop_last=self.drop_last,
        )
        return loader
    
    def __str__(self) -> str:
        return 'ham10000'