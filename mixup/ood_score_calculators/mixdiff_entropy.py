import torch
from torch.distributions.categorical import Categorical

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_logit_based import MixDiffLogitBasedMixin


class MixDiffEntropy(
    MixDiffLogitBasedMixin,
    OODScoreCalculator,
):
    name = 'entropy'

    @torch.no_grad()
    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        known_logits = known_logits.float()
        known_probs = torch.softmax(known_logits, dim=-1)
        known_entropy = Categorical(known_probs).entropy()

        unknown_logits = unknown_logits.float()
        unknown_probs = torch.softmax(unknown_logits, dim=-1)
        unknown_entropy = Categorical(unknown_probs).entropy()

        mixdiff = unknown_entropy - known_entropy
        return mixdiff

    @torch.no_grad()
    def calculate_base_scores(
        self,
        logits,
        **kwargs,
    ):
        if self.add_base_score:
            logits = logits.float()
            probs = torch.softmax(logits, dim=-1)
            entropy = Categorical(probs).entropy()
        else:
            entropy = torch.zeros_like(logits[..., -1])
        return entropy