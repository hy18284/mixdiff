import torch
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
import clip

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_logit_based import MixDiffLogitBasedMixin


class MixDiffOneHot(
    MixDiffLogitBasedMixin,
    OODScoreCalculator,
):
    name = 'onehot'
    @torch.no_grad()
    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        # (N, NC)
        mixdiff = -(known_logits * unknown_logits).mean(-1)
        return mixdiff

    @torch.no_grad()
    def calculate_base_scores(
        self,
        logits,
        **kwargs,
    ):
        base_scores = torch.zeros_like(logits[..., -1])
        return base_scores