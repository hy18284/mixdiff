import torch

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_logit_based import MixDiffLogitBasedMixin


class MixDiffMaxLogitScore(
    MixDiffLogitBasedMixin,
    OODScoreCalculator,
):
    name = 'mls'

    @torch.no_grad()
    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        known_max, _ = torch.max(known_logits, dim=-1) 
        unknown_max, _ = torch.max(unknown_logits, dim=-1) 

        mixdiff = -(unknown_max - known_max)
        return mixdiff

    @torch.no_grad()
    def calculate_base_scores(
        self,
        logits,
        **kwargs,
    ):
        if self.add_base_score:
            max_logits, _ = torch.max(logits, dim=-1)
        else:
            max_logits = torch.zeros_like(logits[..., -1])
        return -max_logits