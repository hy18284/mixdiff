import torch
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
import clip

from .ood_score_calculator import OODScoreCalculator
from .mixdiff_logit_based import MixDiffLogitBasedMixin


class MixDiffMaxSofmaxProb(
    MixDiffLogitBasedMixin,
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
        known_max, _ = torch.max(known_probs, dim=-1) 

        unknown_probs = self._process_logits(unknown_logits)
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
            probs = self._process_logits(logits)
            max_probs, _ = torch.max(probs, dim=-1)
        else:
            max_probs = torch.zeros_like(logits[..., -1])
        return -max_probs
    
    def _process_logits(self, logits):
        if self.intermediate_state == 'logit':
            probs = torch.softmax(logits, dim=-1)
        elif self.intermediate_state == 'softmax':
            probs = logits
        else:
            raise ValueError()
        return probs