import json
import pathlib
import csv
import copy
import json
import re

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


class Snips(Dataset):
    def __init__(
        self,
        mode: str,
        tokenizer_path: str,
        path: str='data/snips/nlu-benchmark/2017-06-custom-intent-engines',
        add_oos: bool=False,
        beautify_intents: bool=True,
        val_ratio: float=0.1,
        seed: int=42,
    ):
        super().__init__()
        self.val_ratio = val_ratio
        self.add_oos = add_oos
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.path = path

        self.data = []
        if mode == 'test':
            pattern = "validate_*.json"
        else:
            pattern = "train_*_full.json"

        self.data = []
        for path in pathlib.Path(self.path).rglob(pattern):
            with open(path, encoding="ISO-8859-1") as f:
                data = json.load(f)
                intent = next(iter(data))
                for line in data[intent]:
                    substrs = line['data']
                    query = ''.join([substr['text'] for substr in substrs])
                    self.data.append((query, intent))

        if mode != 'test':
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
       
        self.intents = [] 
        for path in pathlib.Path(self.path).rglob('train_*_full.json'):
            with open(path, encoding="ISO-8859-1") as f:
                data = json.load(f)
                intent = next(iter(data))
                self.intents.append(intent)
        
        self.intents = list(set(self.intents))
        self.intents.sort()

        if beautify_intents:
            self.intents = [
                ' '.join(re.findall('[A-Z][^A-Z]*', intent))
                for intent in self.intents
            ]
            self.data = [
                (query, ' '.join(re.findall('[A-Z][^A-Z]*', intent)))
                for query, intent in self.data
            ]

        if self.add_oos:
            self.oos_data = CLINIC150(
                mode='test',
                tokenizer_path=tokenizer_path,
                add_oos=True,
                oos_only=True,
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
            query, intent = self.data[idx]
            return {
                'query': query.strip(),
                'label': self.intents.index(intent.strip())
            }
    
    def __len__(self):
        if self.add_oos:
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
    

class SnipsDataModule(LightningDataModule):
    def __init__(
        self, 
        batch_size: int,
        tokenizer_path: str,
        path: str='data/snips/nlu-benchmark/2017-06-custom-intent-engines',
        seed: int=42,
        val_ratio=0.1,
    ):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.tokenizer_path = tokenizer_path
        self.seed = seed
        self.val_ratio = val_ratio

    def setup(self, stage: str) -> None:
        self.train = Snips(
            mode='train', 
            tokenizer_path=self.tokenizer_path,
            add_oos=False,
            beautify_intents=True,
            seed=self.seed,
            val_ratio = self.val_ratio,
            path=self.path,
        )
        self.val = Snips(
            mode='val', 
            tokenizer_path=self.tokenizer_path,
            add_oos=False,
            beautify_intents=True,
            seed=self.seed,
            val_ratio = self.val_ratio,
            path=self.path,
        )
        self.test = Snips(
            mode='test', 
            tokenizer_path=self.tokenizer_path,
            add_oos=False,
            beautify_intents=True,
            seed=self.seed,
            path=self.path,
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