import torch
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
import clip

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_logit_based import MixDiffLogitBasedMixin
from .mixdiff_msp import MixDiffMaxSofmaxProb


class MixMaxDiffMaxSofmaxProb(
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
        # (N, NC)
        known_probs = self._process_logits(known_logits)
        unknown_probs = self._process_logits(unknown_logits)
        # (NC)
        unknown_probs, unknown_max_idx = torch.max(unknown_probs, dim=-1) 
        known_probs = known_probs.gather(1, unknown_max_idx.unsqueeze(1)).squeeze()
        return -(known_probs)

    def __str__(self) -> str:
        # TODO: May not be the greatest way to handle this.
        return super().__str__().replace('mixdiff', 'mixmaxdiff')