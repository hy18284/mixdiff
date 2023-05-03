import random
from collections import defaultdict

from torch.utils.data import DataLoader
import torch

from .base_ood_datamodule import BaseOODDataModule
from text_classification.clinic150_datamodule import CLINIC150


class CLINIC150OODDataset(BaseOODDataModule):
    def __init__(
        self, 
        mode: str,
        data_path: str='data/clinc150/data_full.json',
        tokenizer_path: str = 'roberta-base',
    ):
        self.num_known = 150
        self.mode = mode
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path

        self.dataset = CLINIC150(
            mode=self.mode,
            path=self.data_path,
            tokenizer_path=self.tokenizer_path, 
            add_oos=True,
        )

        self.train_dataset = CLINIC150(
            mode='train',
            path=self.data_path,
            tokenizer_path=self.tokenizer_path, 
            add_oos=False,
        )


    def get_splits(self, n_samples_per_class: int, seed: int):
        yield (
            self.train_dataset.intents,
            torch.tensor(range(len(self.train_dataset.intents))),
            self.sample_given_images(n_samples_per_class, seed)
        )

    def sample_given_images(
        self, 
        n_samples_per_class: int,
        seed: int,
    ):
        label_2_samples = defaultdict(list) 
        for pair in self.train_dataset:
            sample, label = pair['query'], pair['label']
            label_2_samples[label].append(sample)
            
        given_images = []
        for label in range(len(self.train_dataset.intents)):
            images = random.Random(seed).choices(
                label_2_samples[label], 
                k=n_samples_per_class
            )
            given_images.append(images)
        
        return given_images
    
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