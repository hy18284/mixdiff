from typing import (
    Optional,
)

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_embed_based import MixDiffEmbedBasedMixin
from .backbones.base_backbone import BaseBackbone


class MixDiffDML(
    MixDiffEmbedBasedMixin,
    OODScoreCalculator,
):
    name = 'dml'
    def __init__(
        self,
        lmda: float,
        batch_size: int,
        backbone: BaseBackbone,
        utilize_mixup: bool = True,
        add_base_score: bool = True,
        selection_mode: str = 'argmax',
        intermediate_state: str = 'logit',
        oracle_sim_mode: str = 'uniform',
        oracle_sim_temp: float = 1.0,
        log_interval: Optional[int] = None,
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
        self.lmda = lmda

    @torch.no_grad()
    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        known_logits = known_logits.float()
        known_probs = self._process_logits(known_logits)
        known_entropy = Categorical(known_probs).entropy()

        unknown_logits = unknown_logits.float()
        unknown_probs = self._process_logits(unknown_logits)
        unknown_entropy = Categorical(unknown_probs).entropy()

        mixdiff = unknown_entropy - known_entropy
        return mixdiff

    @torch.no_grad()
    def calculate_base_scores(
        self,
        logits,
        embeds,
        class_embeds,
        **kwargs,
    ):
        if self.add_base_score:
            # (N, 1, H) (1, NC, H) -> (N, NC) -> (N)
            max_cos = F.cosine_similarity(embeds.unsqueeze(1), class_embeds.unsqueeze(0), dim=-1)
            max_cos = max_cos.squeeze(0)
            max_cos, _ = torch.max(max_cos, dim=-1)
            # (N, H) -> (N)
            max_norm = torch.norm(embeds, dim=-1, p=2)
            dml = self.lmda * max_cos + max_norm
        else:
            dml = torch.zeros_like(logits[..., -1])
        return -dml

    def _process_logits(self, logits):
        if self.intermediate_state == 'logit':
            probs = torch.softmax(logits, dim=-1)
        elif self.intermediate_state == 'softmax':
            probs = logits
        else:
            raise ValueError()
        return probs