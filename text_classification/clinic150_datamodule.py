import json
import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
)
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer


class CLINIC150(Dataset):
    def __init__(
        self,
        mode: str,
        path: str='data/clinc150/data_full.json',
        tokenizer_path: str = 'roberta-base',
        add_oos: bool=False,
    ):
        super().__init__()
        self.data_path = path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        with open(self.data_path) as f:
            raw_data = json.load(f)
        
        self.intents = [sample[1] for sample in raw_data['train']]
        self.intents = list(set(self.intents))
        self.intents.sort()

        if mode == 'val':
            self.data = raw_data['val']
        elif mode == 'train':
            self.data = raw_data['train']
        elif mode =='test':
            self.data = raw_data['test']
        else:
            ValueError

        if add_oos:
            if mode == 'val':
                self.data += raw_data['oos_val']
            elif mode == 'train':
                self.data += raw_data['oos_train']
            elif mode =='test':
                self.data += raw_data['oos_test']
            else:
                ValueError

            self.intents.append('oos')
        
    def __getitem__(self, idx):
        query, intent = self.data[idx]
        return {
            'query': query.strip(),
            'label': self.intents.index(intent.strip())
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
    

class CLINIC150DataModule(LightningDataModule):
    def __init__(
        self, 
        batch_size: int,
        tokenizer_path: str,
        path: str='data/clinc150/data_full.json',
    ):
        super().__init__()
        self.data_path = path
        self.batch_size = batch_size
        self.tokenizer_path = tokenizer_path

    def setup(self, stage: str) -> None:
        self.train = CLINIC150(mode='train', tokenizer_path=self.tokenizer_path)
        self.val = CLINIC150(mode='val', tokenizer_path=self.tokenizer_path)
        self.test = CLINIC150(mode='test', tokenizer_path=self.tokenizer_path)
    
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