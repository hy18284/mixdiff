import torch

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_logit_based_text import MixDiffLogitBasedMixinText


class MixDiffEnergyText(
    MixDiffLogitBasedMixinText,
    OODScoreCalculator,
):
    name = 'energy'

    @torch.no_grad()
    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        known_logits = known_logits.float()
        unknown_logits = unknown_logits.float()
        known_neg_energy = torch.logsumexp(known_logits, dim=-1)
        unknown_neg_energy = torch.logsumexp(unknown_logits, dim=-1)
        mixdiff = -(unknown_neg_energy - known_neg_energy)
        return mixdiff

    @torch.no_grad()
    def calculate_base_scores(
        self,
        logits,
        **kwargs,
    ):
        if self.add_base_score:
            logits = logits.float()
            neg_energy = torch.logsumexp(logits, dim=-1)
        else:
            neg_energy = torch.zeros_like(logits[..., -1])
        return -neg_energy