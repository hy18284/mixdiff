import itertools
from typing import (
    List,
    Tuple,
    Union,
)

from transformers import (
    AutoModel,
    AutoTokenizer,
)
import torch
import numpy as np

from .base_mixup_operator import BaseMixupOperator


class StringMixup(BaseMixupOperator):
    def __call__(
        self, 
        oracle: List[List[str]], 
        samples: List[str], 
        rates: List[float]
    ) -> Tuple[List[List[List[List[str]]]], List[List[List[str]]]]:
        """Performs mixup and returns oracle, in-batch mixed samples

        Args:
            oracle (List[List[str]]): List of strings with size (N, M).
            samples (List[str]): List of strings with size (N).
            rates (List[str]): List of mixup rates with size (R).

        Returns:
            Tuple[List[List[List[List[str]]]], List[List[List[str]]]]:
            Mixed oracle, in-batch samples each of which has size of
            (N, M, N, R), (N, N, R), respectively.
        """
        rates = rates.tolist()

        masks_list = self._make_mixup_masks(samples, rates)

        # (N * M * N * R) 
        oracle_mixup = []
        for oracle_images in oracle:
            for oracle_image in oracle_images:
                for sample, masks in zip(samples, masks_list):
                    mixup = [
                        self.mixup(oracle_image, sample, mask) 
                        for mask in masks
                    ]
                    oracle_mixup += mixup
        
        # (N * N * R) 
        target_mixup = [] 
        for sample_x in samples:
            for sample_y, masks in zip(samples, masks_list):
                mixup = [
                    self.mixup(sample_x, sample_y, mask) 
                    for mask in masks
                ]
                target_mixup += mixup
        
        return oracle_mixup, target_mixup

    def mixup(self, x: str, y: str, mask: List[int]):
        x = x.split()
        y = y.split()

        mixed = []
        for i in range(len(y)):
            if i < len(x):
                token = y[i] if i in mask else x[i]
                mixed.append(token)
            else:
                if i in mask:
                    mixed.append(y[i])

        mixed = ' '.join(mixed)
        return mixed
    
    def _pad_text(self, oracle, samples):
        texts = itertools.chain.from_iterable(oracle)
        texts += samples
        texts = [text.split() for text in texts]
        max_len = max(texts, key=len)

        padded_samples = []
        for sample in samples:
            sample = sample.split()
            sample += [None] * (max_len - len(sample))
            padded_samples.append(samples)

        padded_oracle_imgs_list = []
        for oracle_images in oracle:
            padded_oracle_imgs = []
            for oracle_image in oracle_images:
                oracle_image = oracle_image.split()
                oracle_image += [None] * (max_len - len(oracle_image))
                padded_oracle_imgs.append(samples)
            padded_oracle_imgs_list.append(padded_oracle_imgs)
        
        return padded_oracle_imgs_list, padded_samples
    
    def _make_mixup_masks(self, samples, rates):
        masks_list = []
        for sample in samples:
            masks = []
            sample = sample.split()
            for rate in rates:
                n_kept = round(len(sample) * (1 - rate))
                mask = np.random.choice(
                    np.arange(len(sample)),
                    n_kept,
                    replace=False,
                )
                masks.append(mask)
            masks_list.append(masks)
        return masks_list

    def __str__(self):
        return 'str'