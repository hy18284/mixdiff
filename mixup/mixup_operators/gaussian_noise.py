from typing import (
    Optional,
    Callable,
    List,
)

import torch
from skimage.util import random_noise

from .base_mixup_operator import BaseMixupOperator


class GaussianNoise(BaseMixupOperator):
    def __init__(self, stds: List[float]):
        self.stds = stds

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

        P, C, H, W = references.size()

        # (P, R, C, H, W)
        stds = torch.tensor(self.stds).to(references.device)
        stds = stds[None, :, None, None, None].expand(P, -1, C, H, W)
        generator = torch.Generator(device=references.device)
        generator = generator.manual_seed(seed)
        noise = torch.normal(mean=0.0, std=stds, generator=generator)
        
        if oracle is not None:
            if oracle.dim() == 5:
                # (B, M, 1, 1, C, H, W) + (1, 1, P, R, C, H, W)
                oracle_mixup = oracle[:, :, None, None, ...] + noise[None, None, ...]
            else:
                # (M, 1, 1, C, H, W) + (1, P, R, C, H, W)
                oracle_mixup = oracle[:, None, None, ...] + noise[None, ...]
                oracle_mixup = torch.clamp(oracle_mixup, min=0.0, max=1.0)
            oracle_mixup = oracle_mixup.reshape(-1, C, H, W)
            if getattr(self, 'post_transform', None) is not None:
                oracle_mixup = self.post_transform(oracle_mixup)

            if targets is None:
                return oracle_mixup
        
        if targets is not None:
            # (B, 1, 1, C, H, W) + (1, P, R, C, H, W)
            target_mixup = targets[:, None, None, ...] + noise[None, ...]
            target_mixup = torch.clamp(target_mixup, min=0.0, max=1.0)
            if getattr(self, 'post_transform', None) is not None:
                target_mixup = self.post_transform(target_mixup)
            target_mixup = target_mixup.reshape(-1, C, H, W)

            if oracle is None:
                return target_mixup
            
        return oracle_mixup, target_mixup
    
    def _add_noise(self, images, n_ref, seed):
        M, C, H, W = images.size()

        noised_images = []
        for image in images:
            diff_refs = []
            for i in range(n_ref):
                diff_rates = []
                for var in self.stds:
                    noised = random_noise(
                        image=image.cpu(), 
                        mode='gaussian', 
                        rng=seed+i,
                        clip=True,
                        mean=0,
                        var=var,
                    )
                    diff_rates.append(torch.tensor(noised).to(image.device))
                diff_refs.append(torch.stack(diff_rates))
            noised_images.append(torch.stack(diff_refs))
        # (M, P, R, C, H, W) -> (M * P * R, C, H, W)
        noised_images = torch.stack(noised_images).view(-1, C, H, W)
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
        return f'gau_{self.stds}'
