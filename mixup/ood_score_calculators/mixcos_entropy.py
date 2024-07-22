import torch
import torch.nn.functional as F

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_entropy import MixDiffEntropy


class MixCosEntropy(
    MixDiffEntropy,
    OODScoreCalculator,
):
    name = 'entropy'
    @torch.no_grad()
    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        known_probs = self._process_logits(known_logits)
        unknown_probs = self._process_logits(unknown_logits)

        # (N, NC), (N, NC) -> (N)
        mixcos = F.cosine_similarity(
            known_probs, 
            unknown_probs, 
            dim=-1,
        )

        return -mixcos

    def __str__(self) -> str:
        # TODO: May not be the greatest way to handle this.
        return super().__str__().replace('mixdiff', 'mixcos')