import torch
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
import clip

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_logit_based import MixDiffLogitBasedMixin


class RandomScore(
    MixDiffLogitBasedMixin,
    OODScoreCalculator,
):
    name = 'rnd_score'
    @torch.no_grad()
    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        bsz = known_logits.size(0)
        return torch.rand(bsz).to(self.device)

    @torch.no_grad()
    def calculate_base_scores(
        self,
        logits,
        **kwargs,
    ):
        bsz = logits.size(0)
        return torch.rand(bsz).to(self.device)