import json
import pathlib
import csv
import copy
from typing import (
    Optional,
)

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from datasets import load_dataset

from .clinic150_datamodule import CLINIC150


class Banking77(Dataset):
    def __init__(
        self,
        mode: str,
        tokenizer_path: str,
        oos_data: Optional[str]=None,
        beautify_intents: bool=True,
        val_ratio: float=0.1,
        seed: int=42,
    ):
        super().__init__()
        self.val_ratio = val_ratio
        self.oos_data = oos_data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        if mode == 'test':
            self.data = load_dataset('PolyAI/banking77', split='test')
        else:
            self.data = load_dataset('PolyAI/banking77', split='train')
            gen = torch.Generator().manual_seed(seed)
            val, train = random_split(
                self.data, 
                [self.val_ratio, 1 - self.val_ratio],
                generator=gen,
            )
            if mode == 'train':
                self.data = train
            else:
                self.data = val
        
        dataset = load_dataset('PolyAI/banking77', split='train')
        self.intents = copy.deepcopy(dataset.features['label'].names)

        if beautify_intents:
            self.intents = [
                ' '.join(intent.split('_'))
                for intent in self.intents
            ]

        if self.oos_data is not None:
            self.oos_data = CLINIC150(
                mode='test',
                tokenizer_path=tokenizer_path,
                add_oos=True,
                oos_only=True,
                beautify_intents=beautify_intents,
                wiki_for_test=self.oos_data == 'clinic_wiki',
            )
            self.intents += self.oos_data.intents

    def __getitem__(self, idx):
        if idx >= len(self.data):
            query, intent = self.oos_data[idx - len(self.data)]
            return {
                'query': query.strip(),
                'label': len(self.data)
            }
        else:
            pair = self.data[idx]
            query, label = pair['text'], pair['label']
            return {
                'query': query.strip(),
                'label': label
            }
    
    def __len__(self):
        if self.oos_data:
            return len(self.data) + len(self.oos_data)
        else:
            return len(self.data)

    def collate_fn(self, samples):
        labels = [sample['label'] for sample in samples]
        labels = torch.tensor(labels)

        queries = [sample['query'] for sample in samples]
        output = self.tokenizer(
            queries,
            add_special_tokens=True,
            padding=True,
            return_tensors='pt',
            return_attention_mask=True,
        )
        output['labels'] = labels
        return output
    

class Banking77DataModule(LightningDataModule):
    def __init__(
        self, 
        batch_size: int,
        tokenizer_path: str,
        seed: int=42,
        val_ratio=0.1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer_path = tokenizer_path
        self.seed = seed
        self.val_ratio = val_ratio

    def setup(self, stage: str) -> None:
        self.train = Banking77(
            mode='train', 
            tokenizer_path=self.tokenizer_path,
            oos_data=False,
            beautify_intents=True,
            seed=self.seed,
            val_ratio = self.val_ratio,
        )
        self.val = Banking77(
            mode='val', 
            tokenizer_path=self.tokenizer_path,
            oos_data=False,
            beautify_intents=True,
            seed=self.seed,
            val_ratio = self.val_ratio,
        )
        self.test = Banking77(
            mode='test', 
            tokenizer_path=self.tokenizer_path,
            oos_data=False,
            beautify_intents=True,
            seed=self.seed,
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.train.collate_fn,
            num_workers=4,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=self.val.collate_fn,
            num_workers=4,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=self.test.collate_fn,
            num_workers=4,
            shuffle=False,
        )