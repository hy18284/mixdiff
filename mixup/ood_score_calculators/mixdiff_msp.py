import torch
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
import clip

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_logit_based import MixDiffLogitBasedMixin


class MixDiffMaxSofmaxProb(
    MixDiffLogitBasedMixin,
    OODScoreCalculator,
):
    @torch.no_grad()
    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        known_probs = torch.softmax(known_logits, dim=-1)
        known_max, _ = torch.max(known_probs, dim=-1) 

        unknown_probs = torch.softmax(unknown_logits, dim=-1)
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
            probs = torch.softmax(logits, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
        else:
            max_probs = torch.zeros_like(logits[..., -1])
        return -max_probs

    def __str__(self) -> str:
        if not self.utilize_mixup:
            return 'msp'
        if self.calculate_base_scores:
            return 'mixdiff_msp+'
        else:
            return 'mixdiff_msp'