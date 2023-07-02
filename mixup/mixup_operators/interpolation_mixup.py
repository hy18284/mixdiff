from typing import (
    List,
    Tuple,
    Optional,
)

import torch

from .base_mixup_operator import BaseMixupOperator


class InterpolationMixup(BaseMixupOperator):
    def __init__(self):
        pass

    def __call__(
        self, 
        references: torch.FloatTensor,
        rates: torch.FloatTensor,
        oracle: Optional[torch.FloatTensor] = None,
        targets: torch.FloatTensor = None, 
        seed: Optional[int] = None,
    ):
        if oracle is not None:
            oracle_mixup = self._oracle_mixup(
                oracle, references, rates,
            )

            if targets is None:
                return oracle_mixup
        
        if targets is not None:
            target_mixup = self._target_mixup(targets, rates)

            if oracle is None:
                return target_mixup
            
        return oracle_mixup, target_mixup

    def _oracle_mixup(self, chosen_images, images, rates):
        N, M, C, H, W = chosen_images.size()
        R = rates.size(0)

        # (N, M, C, H, W) -> (N, N, M, C, H, W)
        chosen_images = chosen_images.expand(N, -1, -1, -1, -1, -1)
        # (N, N, M, C, H, W) -> (M, N, N, C, H, W)
        chosen_images = chosen_images.permute(2, 1, 0, 3, 4, 5)

        # (N, C, H, W) -> (M, N, N, C, H, W)
        images_m = images.expand(M, N, -1, -1, -1, -1)

        # (M, N, N, C, H, W, 1) * (R) -> (M, N, N, C, H, W, R)
        chosen_images = chosen_images.unsqueeze(-1) * rates 
        images_m = images_m.unsqueeze(-1) * (1 - rates)
        known_mixup = chosen_images + images_m
        del chosen_images
        del images_m
        # (M, N, N, C, H, W, R) -> (N, M, N, R, C, H, W)
        known_mixup = known_mixup.permute(1, 0, 2, 6, 3, 4, 5)
        known_mixup = known_mixup.reshape(-1, C, H, W)
        
        return known_mixup
    
    def _target_mixup(self, images, rates):
        N, C, H, W = images.size()
        R = rates.size(0)

        # (N, C, H, W) -> (N, N, C, H, W) -> (N, N, C, H, W, R)
        images_n_1 = images.expand(N, -1, -1, -1, -1)
        images_n_1 = images_n_1.permute(1, 0, 2, 3, 4).unsqueeze(-1) * rates
        images_n_2 = images.expand(N, -1, -1, -1, -1).unsqueeze(-1) * (1 - rates)
        # (N, C, H, W) -> (M, N, C, H, W)
        unknown_mixup = images_n_1 + images_n_2
        unknown_mixup = unknown_mixup.permute(0, 1, 5, 2, 3, 4)
        unknown_mixup = unknown_mixup.reshape(N * N * R, C, H, W)

        return unknown_mixup

    def __str__(self):
        return ''
