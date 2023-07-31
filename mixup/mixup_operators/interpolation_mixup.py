from typing import (
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
            if oracle.dim() == 5:
                oracle_mixup_list = []
                for orc in oracle:
                    oracle_mixup = self._mixup(
                        orc, references, rates,
                    )
                    oracle_mixup_list.append(oracle_mixup)
                oracle_mixup = torch.cat(oracle_mixup_list, dim=0)
            else:
                oracle_mixup = self._mixup(
                    oracle, references, rates,
                )

            if targets is None:
                return oracle_mixup
        
        if targets is not None:
            target_mixup = self._mixup(targets, references, rates)

            if oracle is None:
                return target_mixup
            
        return oracle_mixup, target_mixup
    
    def _mixup(self, images, ref_images, rates):
        # Assuming they are oracle-ref mixup, the dimensions are:
        # (M, C, H, W), (P, C, H, W) -> (M, P, R, C, H, W)

        M, C, H, W = images.size()
        P = ref_images.size(0)
        R = rates.size(0)

        # (M, C, H, W) -> (M, P, C, H, W)
        images = images.unsqueeze(1)
        images = images.expand(-1, P, -1, -1, -1)

        # (P, C, H, W) -> (M, P, C, H, W)
        ref_images = ref_images.expand(M, -1, -1, -1, -1)

        # (M, P, C, H, W, 1) * (R) -> (M, P, C, H, W, R)
        images = images.unsqueeze(-1) * rates 
        ref_images = ref_images.unsqueeze(-1) * (1 - rates)
        known_mixup = images + ref_images
        del images
        del ref_images
        # (M, P, C, H, W, R) -> (M, P, R, C, H, W) -> (M * P * R, C, H, W)
        known_mixup = known_mixup.permute(0, 1, 5, 2, 3, 4)
        known_mixup = known_mixup.reshape(-1, C, H, W)
        
        return known_mixup
    
    def __str__(self):
        return ''
