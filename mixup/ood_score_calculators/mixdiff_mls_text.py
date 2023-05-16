import torch

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_logit_based_text import MixDiffLogitBasedMixinText
from .mixdiff_logit_based_text_zs import MixDiffLogitBasedMixinTextZS


class MixDiffMaxLogitScoreText(
    MixDiffLogitBasedMixinText,
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

        
class MixDiffMaxLogitScoreTextZS(
    MixDiffLogitBasedMixinTextZS,
    OODScoreCalculator,
):
    name = 'mls_zs'

    @torch.no_grad()
    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        # (N * P * R, NC * 2) -> 
        known_logits = known_logits.view(-1, self.NC, 2)
        known_max = self._get_max_logits(known_logits)
        # (N * P * R, NC, 2) -> 
        unknown_max = self._get_max_logits(unknown_logits)

        # (N * P * R)
        mixdiff = -(unknown_max - known_max)
        return mixdiff

    def _get_max_logits(self, logits):
        B = logits.size(0)
        # (N * P * R, NC * 2) -> (N * P * R, NC, 2)
        logits = logits.view(B, -1, 2)
        # (N * P * R, NC, 2) -> (N * P * R, NC)
        logits = logits[:, :, 0] - logits[:, :, 1]
        # (N * P * R, NC) -> (N * P * R)
        max_logits, _ = torch.max(logits, dim=-1) 
        return max_logits

    @torch.no_grad()
    def calculate_base_scores(
        self,
        logits,
        **kwargs,
    ):
        if self.add_base_score:
            max_logits = self._get_max_logits(logits)
        else:
            max_logits = torch.zeros_like(logits[..., -1])
        return -max_logits
