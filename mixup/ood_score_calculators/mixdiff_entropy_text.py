import torch
from torch.distributions.categorical import Categorical

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_logit_based_text import MixDiffLogitBasedMixinText


class MixDiffEntropyText(
    MixDiffLogitBasedMixinText,
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
        known_probs = self._convert_to_probs(known_logits)
        known_entropy = Categorical(known_probs).entropy()

        unknown_logits = unknown_logits.float()
        unknown_probs = self._convert_to_probs(unknown_logits)
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
            probs = self._convert_to_probs(logits)
            entropy = Categorical(probs).entropy()
        else:
            entropy = torch.zeros_like(logits[..., -1])
        return entropy
    
    def _convert_to_probs(self, logits):
        if self.intermediate_state == 'logit':
            probs = torch.softmax(logits, dim=-1)
        elif self.intermediate_state == 'softmax':
            probs = logits
        else:
            raise ValueError
        return probs