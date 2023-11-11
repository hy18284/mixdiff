import torch

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_logit_based_text_ood_class import MixDiffLogitBasedMixinText


class MixDiffOODClass(
    MixDiffLogitBasedMixinText,
    OODScoreCalculator,
):
    name = 'ood_cls'
    @torch.no_grad()
    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        known_scores = self._process_logits(known_logits)
        unknown_scores = self._process_logits(unknown_logits)
        mixdiff = unknown_scores - known_scores
        return mixdiff

    @torch.no_grad()
    def calculate_base_scores(
        self,
        logits,
        **kwargs,
    ):
        if self.add_base_score:
            scores = self._process_logits(logits)
        else:
            scores  = torch.zeros_like(logits[..., -1])
        return scores
    
    def _process_logits(self, logits):
        return logits[..., -1]
