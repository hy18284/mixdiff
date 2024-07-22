from typing import (
    Optional,
    Callable,
)

import torch
import wandb

from .base_mixup_operator import BaseMixupOperator


def mask_ref(references, ratio):
    H, W, = references.size()[-2:]
    h_mask = torch.logical_and(torch.arange(H) > H * ratio, torch.arange(H) < H * (1.0 - ratio))
    # print(h_mask)
    w_mask = torch.logical_and(torch.arange(W) > W / ratio, torch.arange(W) < W * (1.0 - ratio))
    mask = torch.ones(size=(H, W), dtype=bool)
    mask = torch.logical_and(mask, h_mask[:, None])
    mask = torch.logical_and(mask, w_mask[None, :])
    # print(mask, mask.size())
    references = references.masked_fill(mask.to(references.device), 0.5)
    # wandb.log({'debug_ref': wandb.Image(references[1])})
    # wandb.log({'debug_ref': wandb.Image(references[5])})
    return references


class PadedInterpolationMixup(BaseMixupOperator):
    def __init__(self, ratio: float = 0.25):
        super().__init__()
        self.ratio = ratio

    def __call__(
        self, 
        references: torch.FloatTensor,
        rates: torch.FloatTensor,
        oracle: Optional[torch.FloatTensor] = None,
        targets: torch.FloatTensor = None, 
        seed: Optional[int] = None,
    ):
        if oracle is not None:
            orig_oracle_dim = oracle.dim()
        
        if references is not None: 
            references = mask_ref(references, self.ratio)

        if getattr(self, 'pre_transform', None) is not None:
            oracle = self.pre_transform(oracle) if oracle is not None else oracle
            references = self.pre_transform(references) if references is not None else references
            targets = self.pre_transform(targets) if targets is not None else targets

        if oracle is not None:
            if orig_oracle_dim == 5:
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
            if getattr(self, 'post_transform', None) is not None:
                oracle_mixup = self.post_transform(oracle_mixup)

            if targets is None:
                return oracle_mixup
        
        if targets is not None:
            target_mixup = self._mixup(targets, references, rates)
            if getattr(self, 'post_transform', None) is not None:
                target_mixup = self.post_transform(target_mixup)

            if oracle is None:
                return target_mixup
            
        return oracle_mixup, target_mixup
    
    # def _mixup(self, images, ref_images, rates):
    #     # Assuming they are oracle-ref mixup, the dimensions are:
    #     # (M, C, H, W), (P, C, H, W) -> (M, P, R, C, H, W)

    #     M, C, H, W = images.size()
    #     P = ref_images.size(0)
    #     R = rates.size(0)

    #     # (M, C, H, W) -> (M, P, C, H, W)
    #     images = images.unsqueeze(1)
    #     images = images.expand(-1, P, -1, -1, -1)

    #     # (P, C, H, W) -> (M, P, C, H, W)
    #     ref_images = ref_images.expand(M, -1, -1, -1, -1)

    #     # (M, P, C, H, W, 1) * (R) -> (M, P, C, H, W, R)
    #     images = images.unsqueeze(-1) * rates 
    #     ref_images = ref_images.unsqueeze(-1) * (1 - rates)
    #     known_mixup = images + ref_images
    #     del images
    #     del ref_images
    #     # (M, P, C, H, W, R) -> (M, P, R, C, H, W) -> (M * P * R, C, H, W)
    #     known_mixup = known_mixup.permute(0, 1, 5, 2, 3, 4)
    #     known_mixup = known_mixup.reshape(-1, C, H, W)
        
    #     return known_mixup

    def _mixup(self, images, ref_images, rates):
        # Assuming they are oracle-ref mixup, the dimensions are:
        # (M, C, H, W), (P, C, H, W) -> (M, P, R, C, H, W)
        M, *sz_rest = images.size()
        P = ref_images.size(0)
        R = rates.size(0)

        # (M, C, H, W) ->  (M, C * H * W) -> (M, P, C * H * W)
        images = images.view(M, 1, -1)
        images = images.expand(-1, P, -1)

        # (P, C, H, W) -> (M, P, C * H * W)
        ref_images = ref_images.view(1, P, -1)
        ref_images = ref_images.expand(M, -1, -1)

        # (M, P, C * H * W, 1) * (R) -> (M, P, C * H * W, R)
        images = images.unsqueeze(-1) * rates 
        ref_images = ref_images.unsqueeze(-1) * (1 - rates)
        known_mixup = images + ref_images
        del images
        del ref_images
        # (M, P, C * H * W, R) -> (M, P, R, C, H, W) -> (M * P * R, C, H, W)
        known_mixup = known_mixup.view(M, P, *sz_rest, R)
        dim_r_idx = known_mixup.dim() - 1
        idx_rest = list(range(2, len(sz_rest) + 2))
        known_mixup = known_mixup.permute(0, 1, dim_r_idx, *idx_rest)
        known_mixup = known_mixup.reshape(-1, *sz_rest)
        
        return known_mixup
    
    def __str__(self):
        return ''
