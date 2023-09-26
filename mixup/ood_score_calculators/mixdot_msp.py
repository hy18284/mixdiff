import torch
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
import clip

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_logit_based import MixDiffLogitBasedMixin
from .mixdiff_msp import MixDiffMaxSofmaxProb


class MixDotMaxSoftmaxProb(
    MixDiffMaxSofmaxProb,
    OODScoreCalculator,
):
    name = 'msp'
    @torch.no_grad()
    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        known_probs = self._process_logits(known_logits)
        unknown_probs = self._process_logits(unknown_logits)
        mixsim = -(known_probs * unknown_probs).sum(-1)

        return mixsim

    def __str__(self) -> str:
        # TODO: May not be the greatest way to handle this.
        return super().__str__().replace('mixdiff', 'mixsim')