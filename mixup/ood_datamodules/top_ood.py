import random
from collections import defaultdict

from torch.utils.data import DataLoader
import torch

from .base_ood_datamodule import BaseOODDataModule
from text_classification.top_datamodule import Top


class TopOODDataset(BaseOODDataModule):
    def __init__(
        self, 
        mode: str,
        data_path: str='data/top/top-dataset-semantic-parsing',
        tokenizer_path: str = 'bert-base-uncased',
        beautify_intents: bool=True,
    ):
        self.mode = mode
        self.tokenizer_path = tokenizer_path
        self.data_path = data_path
        
        self.dataset = Top(
            mode=self.mode,
            path=self.data_path,
            tokenizer_path=self.tokenizer_path, 
            add_oos=True,
            beautify_intents=beautify_intents,
        )

        self.train_dataset = Top(
            mode='train',
            path=self.data_path,
            tokenizer_path=self.tokenizer_path, 
            add_oos=None,
            beautify_intents=beautify_intents,
        )


    def get_splits(
        self, 
        n_samples_per_class: int, 
        seed: int,
        n_ref_samples,
    ):
        given_images = self.sample_given_images(
            n_samples_per_class, 
            seed, 
        )

        if self.ref_mode == 'oracle':
            ref_images = given_images
        elif self.ref_mode == 'rand_id':
            ValueError('rand_id is unsupported')
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
    ):
        label_2_samples = defaultdict(list) 
        for idx in range(len(self.train_dataset)):
            pair = self.train_dataset[idx]
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
            num_workers=4, 
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
        return loader
    
    def __str__(self) -> str:
        return 'top'