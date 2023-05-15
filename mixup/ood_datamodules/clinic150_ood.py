import random
from collections import defaultdict
import itertools

from torch.utils.data import DataLoader
import torch
import numpy as np

from .base_ood_datamodule import BaseOODDataModule
from text_classification.clinic150_datamodule import CLINIC150


class CLINIC150OODDataset(BaseOODDataModule):
    def __init__(
        self, 
        mode: str,
        data_path: str='data/clinc150',
        tokenizer_path: str = 'bert-base-uncased',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path

        self.dataset = CLINIC150(
            mode=self.mode,
            path=self.data_path,
            tokenizer_path=self.tokenizer_path, 
            add_oos=True,
            wiki_for_test=False,
        )

        self.train_dataset = CLINIC150(
            mode='train',
            path=self.data_path,
            tokenizer_path=self.tokenizer_path, 
            add_oos=False,
        )

    def get_splits(
        self, 
        n_samples_per_class: int, 
        seed: int, 
        n_ref_samples,
    ):
        given_images, ref_images = self.sample_given_images(
            n_samples_per_class, 
            seed, 
            n_ref_samples,
        )
        if self.ref_mode == 'oracle':
            ref_images = given_images
        elif self.ref_mode == 'rand_id':
            assert ref_images is not None
        elif self.ref_mode == 'in_batch':
            ref_images = None

        yield (
            self.train_dataset.intents,
            torch.tensor(range(len(self.train_dataset.intents))),
            given_images,
            ref_images,
            None,
        )

    def sample_given_images(
        self, 
        n_samples_per_class: int,
        seed: int,
        n_ref_samples: int = None,
    ):
        label_2_samples = defaultdict(list) 
        for pair in self.train_dataset:
            sample, label = pair['query'], pair['label']
            label_2_samples[label].append(sample)

        if n_ref_samples is not None:
            n_ref_per_class = n_ref_samples / len(label_2_samples)
            n_ref_per_class = int(np.ceil(n_ref_per_class))
        else:
            n_ref_per_class = 0
            
        given_images = []
        for label in range(len(self.train_dataset.intents)):
            images = random.Random(seed).choices(
                label_2_samples[label], 
                k=n_samples_per_class + n_ref_per_class
            )
            given_images.append(images)

        if n_ref_samples is not None:
            ref_images = [
                given[n_samples_per_class:] for given
                in given_images
            ]
            ref_images = list(itertools.chain.from_iterable(ref_images))
            ref_images = random.Random(seed).choices(
                ref_images, 
                k=n_ref_samples
            )
        else:
            ref_images = None
        
        given_images = [
            given[:n_samples_per_class] for given
            in given_images
        ]

        return given_images, ref_images
    
    def construct_loader(self, batch_size: int, shuffle: bool = True):

        def collate_fn(samples):
            return (
                [sample['query'] for sample in samples],
                torch.tensor([sample['label'] for sample in samples]),
            )

        loader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            num_workers=2, 
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
        return loader
    
    def __str__(self) -> str:
        return 'clinic150'


class CLINIC150OODDatasetWiki(CLINIC150OODDataset):
    def __init__(
        self, 
        mode: str,
        data_path: str='data/clinc150',
        tokenizer_path: str = 'bert-base-uncased',
    ):
        super().__init__(mode=mode, data_path=data_path, tokenizer_path=tokenizer_path)
        self.dataset = CLINIC150(
            mode=self.mode,
            path=self.data_path,
            tokenizer_path=self.tokenizer_path, 
            add_oos=True,
            wiki_for_test=True,
        )

    def __str__(self) -> str:
        return 'clinic150_wki'