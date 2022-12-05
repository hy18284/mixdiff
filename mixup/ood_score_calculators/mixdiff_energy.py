import torch

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_logit_based import MixDiffLogitBasedMixin


class MixDiffEnergy(
    MixDiffLogitBasedMixin,
    OODScoreCalculator,
):
    @torch.no_grad()
    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        logit_scale = self.clip_model.logit_scale.exp().float()
        known_logits = known_logits.float()
        unknown_logits = unknown_logits.float()
        known_neg_energy = torch.logsumexp(known_logits, dim=-1) / logit_scale
        unknown_neg_energy = torch.logsumexp(unknown_logits, dim=-1) / logit_scale
        mixdiff = -(unknown_neg_energy - known_neg_energy)
        return mixdiff

    @torch.no_grad()
    def calculate_base_scores(
        self,
        logits,
        **kwargs,
    ):
        if self.add_base_score:
            logit_scale = self.clip_model.logit_scale.exp().float()
            logits = logits.float()
            neg_energy = torch.logsumexp(logits, dim=-1) / logit_scale
        else:
            neg_energy = torch.zeros_like(logits[..., -1])
        return -neg_energy

    def __str__(self) -> str:
        if not self.utilize_mixup:
            return 'energy'
        if self.calculate_base_scores:
            return 'mixdiff_energy+'
        else:
            return 'mixdiff_energy'