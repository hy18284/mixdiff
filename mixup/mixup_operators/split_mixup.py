from typing import (
    List,
    Tuple,
)

import numpy as np

from .base_mixup_operator import BaseMixupOperator


class SplitMixup(BaseMixupOperator):
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

        masks_list, fronts_list = self._make_mixup_masks(samples, rates)

        # (N * M * N * R) 
        oracle_mixup = []
        for oracle_images in oracle:
            for oracle_image in oracle_images:
                for sample, masks, fronts in zip(samples, masks_list, fronts_list):
                    mixup = [
                        self.mixup(oracle_image, sample, mask, front) 
                        for mask, front in zip(masks, fronts)
                    ]
                    oracle_mixup += mixup
        
        # (N * N * R) 
        target_mixup = [] 
        for sample_x in samples:
            for sample_y, masks, fronts in zip(samples, masks_list, fronts_list):
                mixup = [
                    self.mixup(sample_x, sample_y, mask, front) 
                    for mask, front in zip(masks, fronts)
                ]
                target_mixup += mixup
        
        return oracle_mixup, target_mixup

    def mixup(self, x: str, y: str, mask: List[int], front: bool):
        x = x.split()
        y = y.split()

        if not front:
            x = list(reversed(x))
            y = list(reversed(y))

        mixed = []
        for i in range(len(y)):
            if i < len(x):
                token = y[i] if i in mask else x[i]
                mixed.append(token)
            else:
                if i in mask:
                    mixed.append(y[i])

        if not front:
            mixed = reversed(mixed)

        mixed = ' '.join(mixed)
        return mixed
    
    def _make_mixup_masks(self, samples, rates):
        masks_list = []
        fronts_list = []
        for sample in samples:
            masks = []
            fronts = []
            sample = sample.split()
            for rate in rates:
                n_lost = round(len(sample) * (rate))
                mask = list(range(n_lost, len(sample)))
                front = np.random.random() > 0.5
                masks.append(mask)
                fronts.append(front)
            masks_list.append(masks)
            fronts_list.append(fronts)
        return masks_list, fronts_list

    def __str__(self):
        return 'spl'