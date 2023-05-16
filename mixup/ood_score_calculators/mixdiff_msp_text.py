import torch
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
import clip

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_logit_based_text import MixDiffLogitBasedMixinText
from .mixdiff_logit_based_text_zs import MixDiffLogitBasedMixinTextZS


class MixDiffMaxSofmaxProbText(
    MixDiffLogitBasedMixinText,
    OODScoreCalculator,
):
    name = 'msp'
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


class MixDiffMaxSoftmaxProbTextZS(
    MixDiffLogitBasedMixinTextZS,
    OODScoreCalculator,
):
    name = 'msp_zs'

    @torch.no_grad()
    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        # (N * P * 2, NC) -> (N * P, NC, 2)
        known_logits = known_logits.view(-1, self.NC, 2)
        known_max = self._get_max_prob(known_logits)
        # (N * P * R, NC, 2) -> 
        unknown_max = self._get_max_prob(unknown_logits)

        # (N * P * R)
        mixdiff = -(unknown_max - known_max)
        return mixdiff

    def _get_max_prob(self, logits):
        B = logits.size(0)
        # (N * P * R, NC * 2) -> (N * P * R, NC, 2)
        logits = logits.view(B, -1, 2)
        # (N * P * R, NC, 2) -> (N * P * R, NC)
        probs = torch.softmax(logits, dim=-1)[:, :, 0]
        # (N * P * R, NC) -> (N * P * R)
        max_probs, _ = torch.max(probs, dim=-1) 
        return max_probs

    @torch.no_grad()
    def calculate_base_scores(
        self,
        logits,
        **kwargs,
    ):
        if self.add_base_score:
            max_probs = self._get_max_prob(logits)
        else:
            max_probs = torch.zeros_like(logits[..., -1])
        return -max_probs
