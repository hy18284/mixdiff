import random
from collections import defaultdict
import itertools
from jsonargparse import ArgumentParser
from typing import (
    List,
)
import yaml
import copy

from torch.utils.data import (
    DataLoader,
    Dataset,
)
import torch
import numpy as np

from .base_ood_datamodule import BaseOODDataModule
from text_classification.class_split_datamodule import ClassSplitWrapper


class ClassSplitDataConnector:
    def __init__(self, config_path: str):
        self.parser = ArgumentParser()
        self.parser.add_subclass_arguments(Dataset, 'train_dataset')
        self.parser.add_subclass_arguments(Dataset, 'eval_dataset')

        self.parser.add_argument('--test_seeds', type=List[int])
        self.parser.add_argument('--test_id_ratios', type=List[float])
        self.parser.add_argument('--val_seeds', type=List[int], default=[])
        self.parser.add_argument('--val_id_ratios', type=List[float], default=[])

        self.parser.add_argument('--val_ratio', type=float, default=None)
        self.parser.add_argument('--val_split_seed', type=int, default=None)
        self.parser.add_argument('--ood_labels', type=List[int], default=[])

        self.parser.add_argument('--name', type=str)
        self.parser.add_argument('--mode', type=str)
        self.parser.add_argument('--model_path', type=str)

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.mode = self.config['mode']
        
    def iterate_datasets(self):
        if self.mode == 'val':
            for datasets in self._iterate_val_datasets():
                yield datasets
        elif self.mode == 'test':
            for datasets in self._iterate_test_datasets():
                yield datasets
        else:
            raise ValueError('Inavalid mode')
    
    def _iterate_val_datasets(self):
        for test_seed in self.config['test_seeds']:
            for test_id_ratio in self.config['test_id_ratios']:
                for datasets in self._make_val_dataset(
                    test_seed, 
                    test_id_ratio
                ):
                    yield datasets

    
    def _make_val_dataset(self, test_seed, test_id_ratio):
        for val_seed in self.config['val_seeds']:
            for val_id_ratio in self.config['val_id_ratios']:

                config = copy.deepcopy(self.config)

                train_config = config['train_dataset']
                train_config['init_args']['class_split_seed'] = test_seed
                train_config['init_args']['seen_class_ratio'] = test_id_ratio

                args = self.parser.parse_string(yaml.dump(config))
                classes = self.parser.instantiate_classes(args)
                train_dataset = classes['train_dataset']
                eval_dataset = classes['eval_dataset']

                train_dataset = ClassSplitWrapper(
                    dataset=train_dataset,
                    class_split_seed=val_seed,
                    seen_class_ratio=val_id_ratio,
                    ood_labels=args.ood_labels,
                    val_ratio=args.val_ratio,
                    val_split_seed=args.val_split_seed,
                    mode='train'
                )

                dataset = ClassSplitWrapper(
                    dataset=eval_dataset,
                    class_split_seed=val_seed,
                    seen_class_ratio=val_id_ratio,
                    ood_labels=args.ood_labels,
                )

                model_path = args.model_path.format(
                    test_seed,
                    test_id_ratio,
                    val_seed,
                    val_id_ratio,
                )
                yield train_dataset, dataset, model_path

    def _iterate_test_datasets(self):
        for test_seed in self.config['test_seeds']:
            for test_id_ratio in self.config['test_id_ratios']:
                config = copy.deepcopy(self.config)

                args = self.parser.parse_string(yaml.dump(config))
                classes = self.parser.instantiate_classes(args)
                train_dataset = classes['train_dataset']
                eval_dataset = classes['eval_dataset']

                train_dataset = ClassSplitWrapper(
                    dataset=train_dataset,
                    class_split_seed=test_seed,
                    seen_class_ratio=test_id_ratio,
                    ood_labels=args.ood_labels,
                )

                dataset = ClassSplitWrapper(
                    dataset=eval_dataset,
                    class_split_seed=test_seed,
                    seen_class_ratio=test_id_ratio,
                    ood_labels=args.ood_labels,
                )
                model_path = args.model_path.format(
                    test_seed,
                    test_id_ratio,
                )
                yield train_dataset, dataset, model_path

    @property     
    def name(self):
        return self.config['name']


class ClassSplitOODDataset(BaseOODDataModule):
    def __init__(
        self, 
        config_path: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._connector = ClassSplitDataConnector(config_path)

    def get_splits(
        self, 
        n_samples_per_class: int, 
        seed: int, 
        n_ref_samples,
    ):
        for train_dataset, dataset, model_path in self._connector.iterate_datasets():
            self.train_dataset = train_dataset
            self.dataset = dataset

            seen_indices = set()
            for seen_intent in self.train_dataset.intents:
                for idx, intent in enumerate(self.dataset.intents):
                    if seen_intent == intent:
                        seen_indices.add(idx)
            seen_indices = list(seen_indices)

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
                torch.tensor(seen_indices),
                given_images,
                ref_images,
                model_path,
            )
    
    def sample_given_images(
        self, 
        n_samples_per_class: int,
        seed: int,
        n_ref_samples: int = None,
    ):
        label_2_samples = defaultdict(list) 
        for idx in range(len(self.train_dataset)):
            pair = self.train_dataset[idx]
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
            num_workers=4, 
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
        return loader
    
    def __str__(self) -> str:
        return self._connector.name