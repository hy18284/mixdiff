from typing import (
    Optional,
)

import torch

from .ood_score_calculator import OODScoreCalculator
from ..mixup_operators import BaseMixupOperator


class LinearCombinationText(OODScoreCalculator):
    def __init__(
        self,
        base_ood_score_fn: OODScoreCalculator,
        aux_ood_score_fn: OODScoreCalculator,
        gamma: float,
    ):
        self.base_ood_fn = base_ood_score_fn
        self.aux_ood_fn = aux_ood_score_fn
        self.gamma = gamma

        self.utilize_mixup = False
        self.add_base_score = True
    
    def load_model(self, backbone_name, device):
        self.base_ood_fn.load_model(backbone_name=backbone_name, device=device)
        self.aux_ood_fn.load_model(backbone_name=backbone_name, device=device)

    def on_eval_start(
        self, 
        seen_labels, 
        given_images, 
        mixup_fn: Optional[BaseMixupOperator],
        ref_images,
        rates,
        seed,
        iter_idx,
        model_path,
    ):
        self.base_ood_fn.on_eval_start(
            seen_labels=seen_labels,
            given_images=given_images, 
            mixup_fn=mixup_fn,
            ref_images=ref_images,
            rates=rates,
            seed=seed,
            iter_idx=iter_idx,
            model_path=model_path,
        )
        self.aux_ood_fn.on_eval_start(
            seen_labels=seen_labels,
            given_images=given_images, 
            mixup_fn=mixup_fn,
            ref_images=ref_images,
            rates=rates,
            seed=seed,
            iter_idx=iter_idx,
            model_path=model_path,
        )

    def on_eval_end(self, iter_idx: int):
        self.base_ood_fn.on_eval_end(iter_idx=iter_idx)
        self.aux_ood_fn.on_eval_end(iter_idx=iter_idx)

    @torch.no_grad()
    def process_images(
        self,
        images,
    ):
        base_kwargs = self.base_ood_fn.process_images(images=images)
        aux_kwargs = self.aux_ood_fn.process_images(images=images)
        return {
            'kwargs_pair': (base_kwargs, aux_kwargs)
        }
    
    @torch.no_grad()
    def select_given_images(
        self,
        given_images,
        logits,
        **kwargs,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def process_mixed_target(
        self,
        images,
        **kwargs
    ):
        raise NotImplementedError

    @torch.no_grad()
    def process_mixed_oracle(
        self,
        images,
        logits,
        **kwargs
    ):
        raise NotImplementedError
    
    def process_mixup_images(self, images):
        raise NotImplementedError

    def __str__(self) -> str:
        return f'comb_{self.base_ood_fn}_{self.aux_ood_fn}'

    @torch.no_grad()
    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def calculate_base_scores(
        self,
        kwargs_pair,
        **kwargs,
    ):
        base_kwargs, aux_kwargs = kwargs_pair
        base_score = self.base_ood_fn.calculate_base_scores(**base_kwargs)
        base_score = base_score.to(torch.double)
        aux_score = self.aux_ood_fn.calculate_base_scores(**aux_kwargs)
        aux_score = aux_score.to(torch.double)
        scores = base_score + self.gamma * aux_score
        return scores