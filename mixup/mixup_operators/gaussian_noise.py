from typing import (
    Optional,
    Callable,
    List,
)

import torch
from skimage.util import random_noise

from .base_mixup_operator import BaseMixupOperator


class GaussianNoise(BaseMixupOperator):
    def __init__(self, vars: List[float]):
        self.vars = vars

    def __call__(
        self, 
        references: torch.FloatTensor,
        rates: torch.FloatTensor,
        oracle: Optional[torch.FloatTensor] = None,
        targets: torch.FloatTensor = None, 
        seed: Optional[int] = None,
    ):
        if getattr(self, 'pre_transform', None) is not None:
            oracle = self.pre_transform(oracle) if oracle is not None else oracle
            references = self.pre_transform(references) if references is not None else references
            targets = self.pre_transform(targets) if targets is not None else targets

        if oracle is not None:
            if oracle.dim() == 5:
                oracle_mixup_list = []
                for orc in oracle:
                    oracle_mixup = self._add_noise(
                        orc, seed,
                    )
                    oracle_mixup_list.append(oracle_mixup)
                oracle_mixup = torch.cat(oracle_mixup_list, dim=0)
            else:
                oracle_mixup = self._mixup(
                    oracle, references, rates,
                )
            if getattr(self, 'post_transform', None) is not None:
                oracle_mixup = self.post_transform(oracle_mixup)

            if targets is None:
                return oracle_mixup
        
        if targets is not None:
            target_mixup = self._add_noise(targets, seed)
            if getattr(self, 'post_transform', None) is not None:
                target_mixup = self.post_transform(target_mixup)

            if oracle is None:
                return target_mixup
            
        return oracle_mixup, target_mixup
    
    def _add_noise(self, images, seed):
        M, C, H, W = images.size()

        noised_images = []
        for image in images:
            noised_versions = []
            for i, var in enumerate(self.vars):
                noised = random_noise(
                    image=image.cpu(), 
                    mode='gaussian', 
                    rng=seed+i,
                    clip=True,
                    mean=0,
                    var=var,
                )
                noised_versions.append(torch.tensor(noised).to(image.device))
            noised_images.append(torch.stack(noised_versions))
        # (M, P, C, H, W) -> (M, P, R, C, H, W) -> (M * P * R, C, H, W)
        noised_images = torch.stack(noised_images).unsqueeze(2).reshape(-1, C, H, W)
        return noised_images
    
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
        return f'gau_{self.vars}'
