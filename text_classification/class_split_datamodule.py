import json
import torch
import pathlib
import collections
from typing import (
    List,
    Optional,
)

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
import numpy as np
import wandb


class ClassSplitWrapper(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        class_split_seed: int,
        seen_class_ratio: Optional[float] = None,
        ood_labels: List[int] = [],
        val_ratio: Optional[float] = None,
        val_split_seed: Optional[int] = None,
        mode: Optional[str] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.seen_class_ratio = seen_class_ratio

        if val_ratio is not None:
            gen = torch.Generator().manual_seed(val_split_seed)
            val, train = random_split(
                dataset, 
                [val_ratio, 1 - val_ratio],
                generator=gen,
            )
            if mode == 'val':
                dataset = val
            elif mode == 'train':
                dataset = train

        if self.seen_class_ratio is None:
            self.data = dataset
            return
            
        label2samples = collections.defaultdict(list)
        for idx in range(len(dataset)):
            pair = dataset[idx]
            sample, label = pair['query'], pair['label']
            label2samples[label] = sample
        
        self.seen_labels = [
            label for label in range(len(self.dataset.intents))
            if label not in ood_labels
        ]
        total_n_labels = len(self.seen_labels) + len(ood_labels)

        n_seen = round(total_n_labels * seen_class_ratio)
        if n_seen == 0:
            raise ValueError()

        rng = np.random.default_rng(class_split_seed)
        self.seen_labels.sort()
        self.seen_labels = rng.choice(self.seen_labels, n_seen, replace=False)
        self.seen_labels = list(self.seen_labels)
        self.seen_labels.sort()

        self.data = []
        for idx in range(len(dataset)):
            pair = dataset[idx]
            sample, label = pair['query'], pair['label']
            if label in self.seen_labels:
                pair['label'] = self.seen_labels.index(label)
                self.data.append(pair)

        print('Seen labels', self.seen_labels)
        print('# of seen labels:', len(self.seen_labels))

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
    @property 
    def intents(self):
        return [
            self.dataset.intents[label]
            for label in self.seen_labels
        ]
    
    def collate_fn(self, *args, **kwargs):
        return self.dataset.collate_fn(
            *args, **kwargs
        )

        
class ClassSplitDataModule(LightningDataModule):
    def __init__(
        self, 
        batch_size: int,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        class_split_seed: Optional[int] = None,
        seen_class_ratio: Optional[float] = None,
        ood_labels: List[str] = [],
        val_ratio: Optional[float] = None,
        val_split_seed: Optional[int] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.calss_split_seed = class_split_seed
        self.seen_class_ratio = seen_class_ratio
        self.ood_labels = ood_labels

        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train = ClassSplitWrapper(
                dataset=self.train_dataset,
                class_split_seed=self.calss_split_seed,
                seen_class_ratio=self.seen_class_ratio,
                ood_labels=self.ood_labels,
                val_ratio=self.val_ratio,
                val_split_seed=self.val_split_seed,
                mode='train'
            )
            self.val = ClassSplitWrapper(
                dataset=self.val_dataset,
                class_split_seed=self.calss_split_seed,
                seen_class_ratio=self.seen_class_ratio,
                ood_labels=self.ood_labels,
                val_ratio=self.val_ratio,
                val_split_seed=self.val_split_seed,
                mode='val'
            )

            wandb.config['train_intents'] = self.train.intents
            wandb.config['val_intents'] = self.val.intents
            print('train intents', self.train.intents)
            print('val intents', self.val.intents)

        else:
            if self.test_dataset is not None:
                self.test = ClassSplitWrapper(
                    dataset=self.test_dataset,
                    class_split_seed=None,
                    seen_class_ratio=None,
                    ood_labels=None,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=4,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=4,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=4,
            shuffle=False,
        )