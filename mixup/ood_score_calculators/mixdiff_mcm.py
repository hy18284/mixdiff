from typing import (
    Optional,
)

import torch
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
import clip

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_logit_based import MixDiffLogitBasedMixin
from .backbones.base_backbone import BaseBackbone


class MixDiffMCM(
    MixDiffLogitBasedMixin,
    OODScoreCalculator,
):
    name = 'mcm'

    def __init__(
        self,
        batch_size: int,
        backbone: BaseBackbone,
        utilize_mixup: bool = True,
        add_base_score: bool = True,
        selection_mode: str = 'argmax',
        intermediate_state: str = 'logit',
        oracle_sim_mode: str = 'uniform',
        oracle_sim_temp: float = 1.0,
        log_interval: Optional[int] = None,
        temperature: float = 1.0
    ):
        super().__init__(
            batch_size,
            backbone,
            utilize_mixup,
            add_base_score,
            selection_mode,
            intermediate_state,
            oracle_sim_mode,
            oracle_sim_temp,
            log_interval,
        )
        self.temperature = temperature

    @torch.no_grad()
    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        known_probs = self._process_logits(known_logits)
        known_max, _ = torch.max(known_probs, dim=-1) 

        unknown_probs = self._process_logits(unknown_logits)
        unknown_max, _ = torch.max(unknown_probs, dim=-1) 

        mixdiff = -(unknown_max - known_max)
        return mixdiff

    @torch.no_grad()
    def calculate_base_scores(
        self,
        logits,
        **kwargs,
    ):
        if self.add_base_score:
            probs = self._process_logits(logits)
            max_probs, _ = torch.max(probs, dim=-1)
        else:
            max_probs = torch.zeros_like(logits[..., -1])
        return -max_probs
    
    def _process_logits(self, logits):
        if self.intermediate_state == 'logit':
            # Undo scaling.
            logit_scale = self.backbone.clip_model.logit_scale.exp()
            logits = logits / logit_scale
            logits = logits / self.temperature
            probs = torch.softmax(logits, dim=-1)
        else:
            raise ValueError()
        return probs

    @torch.no_grad()
    def process_images(
        self,
        images,
    ):
        kwargs = super().process_images(images)
        probs = self._process_logits(kwargs['logits']) 
        # (N, NC) -> (N)
        max_probs, _ = torch.max(probs, dim=-1)
        scale = max_probs / (1 - 1 / probs.size(1))
        kwargs['scale'] = scale
        # print('scale', scale)
        # print('probs', probs)
        return kwargs
        