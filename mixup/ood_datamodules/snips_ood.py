import random
from collections import defaultdict

from torch.utils.data import DataLoader
import torch

from .base_ood_datamodule import BaseOODDataModule
from text_classification.snips_datamodule import Snips


class SnipsOODDatasetClinicTest(BaseOODDataModule):
    def __init__(
        self, 
        mode: str,
        data_path: str='data/snips/nlu-benchmark/2017-06-custom-intent-engines',
        tokenizer_path: str = 'bert-base-uncased',
        beautify_intents: bool=True,
        val_ratio: float=0.1,
        seed: int=42,
    ):
        self.mode = mode
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        

        self.dataset = Snips(
            mode=self.mode,
            path=self.data_path,
            tokenizer_path=self.tokenizer_path, 
            oos_data='clinic_test',
            beautify_intents=beautify_intents,
            val_ratio=val_ratio,
            seed=seed,
        )

        self.train_dataset = Snips(
            mode='train',
            path=self.data_path,
            tokenizer_path=self.tokenizer_path, 
            oos_data=None,
            beautify_intents=beautify_intents,
            val_ratio=val_ratio,
            seed=seed,
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
        return 'snips_clntst'


class SnipsOODDatasetClinicWiki(SnipsOODDatasetClinicTest):
    def __init__(
        self, 
        mode: str,
        data_path: str='data/snips/nlu-benchmark/2017-06-custom-intent-engines',
        tokenizer_path: str = 'bert-base-uncased',
        beautify_intents: bool=True,
        val_ratio: float=0.1,
        seed: int=42,
    ):
        super().__init__(
            mode=mode,
            data_path=data_path,
            tokenizer_path=tokenizer_path,
            beautify_intents=beautify_intents,
            val_ratio=val_ratio,
            seed=seed,
        )

        self.dataset = Snips(
            mode=self.mode,
            path=self.data_path,
            tokenizer_path=self.tokenizer_path, 
            oos_data='clinic_wiki',
            beautify_intents=beautify_intents,
            val_ratio=val_ratio,
            seed=seed,
        )

    def __str__(self) -> str:
        return 'snips_clnwki'