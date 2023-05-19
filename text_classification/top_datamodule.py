from typing import (
    List,
)
import pathlib
import csv

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer


class Top(Dataset):
    oos_names = [
        'IN:UNSUPPORTED_NAVIGATION', 
        'IN:UNSUPPORTED', 
        'IN:UNSUPPORTED_EVENT',
    ]

    def __init__(
        self,
        mode: str,
        tokenizer_path: str,
        path: str = 'data/top/top-dataset-semantic-parsing',
        add_oos: bool = False,
        beautify_intents: bool = True,
        oos_names: List[str] = [],
    ):
        super().__init__()
        self.data_path = path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.oos_names += oos_names

        if mode == 'train' or mode == 'test':
            file_name = f'{mode}.tsv'
        elif mode == 'val':
            file_name = 'eval.tsv'
        else:
            raise ValueError()
            
        self.data = []
        with open(pathlib.Path(self.data_path) / file_name) as f:
            for line in csv.reader(f, delimiter='\t'):
                query, query_tok, intent = line
                intent = intent.split()
                intent = intent[0][1:]
                if not (intent in self.oos_names) or add_oos:
                    self.data.append((query, intent))
        
        self._intents = []
        with open(pathlib.Path(self.data_path) / 'train.tsv') as f:
            for line in csv.reader(f, delimiter='\t'):
                query, query_tok, intent = line
                intent = intent.split()
                intent = intent[0][1:]
                self._intents.append(intent)
        
        self._intents = list(set(self._intents) - set(self.oos_names))
        self._intents.sort()

        if add_oos:
            self._intents += self.oos_names
        
        if beautify_intents:
            self._intents = [
                ' '.join(intent[3:].split('_'))
                for intent in self._intents
            ]
            self.data = [
                (query, ' '.join(intent[3:].split('_')))
                for query, intent in self.data
            ]
        
        from collections import Counter
        from pprint import pprint
        intents = [
            intent
            for query, intent in self.data
        ]
        print(mode)
        pprint(Counter(intents))
        pprint(len(Counter(intents)))
        

    def __getitem__(self, idx):
        query, intent = self.data[idx]
        return {
            'query': query.strip(),
            'label': self._intents.index(intent.strip())
        }
    
    def __len__(self):
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
    
    @property 
    def intents(self):
        return self._intents
    

class TopDataModule(LightningDataModule):
    def __init__(
        self, 
        batch_size: int,
        tokenizer_path: str,
        path: str='data/top/top-dataset-semantic-parsing',
    ):
        super().__init__()
        self.data_path = path
        self.batch_size = batch_size
        self.tokenizer_path = tokenizer_path

    def setup(self, stage: str) -> None:
        self.train = Top(
            mode='train', 
            tokenizer_path=self.tokenizer_path,
            path=self.data_path,
            add_oos=False,
            beautify_intents=True,
        )
        self.val = Top(
            mode='val', 
            tokenizer_path=self.tokenizer_path,
            path=self.data_path,
            add_oos=False,
            beautify_intents=True,
        )
        self.test = Top(
            mode='test', 
            tokenizer_path=self.tokenizer_path,
            path=self.data_path,
            add_oos=False,
            beautify_intents=True,
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