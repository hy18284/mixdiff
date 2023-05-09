import itertools
from typing import (
    List,
    Tuple,
    Union,
)

from transformers import (
    AutoModel,
    AutoTokenizer,
)
import torch
import numpy as np

from .base_mixup_operator import BaseMixupOperator


class CutMixup(BaseMixupOperator):
    def __init__(
        self, 
        similarity: str='dot',
        model_path: str='roberta-base',
        device: Union[int, str]=0,
    ):
        self.similarity = similarity
        self.device = torch.device(device)
        model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.embedding = model.get_input_embeddings()
        self.embedding.to(self.device)

    def __call__(
        self, 
        oracle: List[List[str]], 
        samples: List[str], 
        rates: List[float]
    ) -> Tuple[List[List[List[List[str]]]], List[List[List[str]]]]:
        """Performs mixup and returns oracle, in-batch mixed samples

        Args:
            oracle (List[List[str]]): List of strings with size (N, M).
            samples (List[str]): List of strings with size (N).
            rates (List[str]): List of mixup rates with size (R).

        Returns:
            Tuple[List[List[List[List[str]]]], List[List[List[str]]]]:
            Mixed oracle, in-batch samples each of which has size of
            (N, M, N, R), (N, N, R), respectively.
        """
        rates = rates.tolist()

        indices_list = self._make_mixup_indices(samples, rates)

        # (N * M * N * R) 
        oracle_mixup = []
        for oracle_images in oracle:
            for oracle_image in oracle_images:
                for sample, indices in zip(samples, indices_list):
                    mixup = [
                        self.mixup(oracle_image, sample, idx) 
                        for idx in indices
                    ]
                    oracle_mixup += mixup
        
        # (N * N * R) 
        target_mixup = [] 
        for sample_x in samples:
            for sample_y, indices in zip(samples, indices_list):
                mixup = [
                    self.mixup(sample_x, sample_y, idx) 
                    for idx in indices
                ]
                target_mixup += mixup
        
        return oracle_mixup, target_mixup

    def mixup(self, x: str, y: str, idx: List[int]):
        x_input_ids = self.tokenizer(
            x, 
            add_special_tokens=False, 
            padding=False, 
            return_tensors='pt',
        )['input_ids'].to(self.device)[0]

        y_input_ids = self.tokenizer(
            ' '.join(y.split()[idx]), 
            add_special_tokens=False, 
            padding=False, 
            return_tensors='pt',
        )['input_ids'].to(self.device)[0]

        y_dense = self.embedding(y_input_ids)
        x_dense = self.embedding(x_input_ids)

        P = y_dense.size(0)
        L = x_input_ids.size(0)

        if L < P:
            x_patch = x
        else:
            # (L) -> (B, P)
            s = torch.arange(x_dense.size(0))
            s = s.unfold(0, P, 1)
            # (L, H) -> (B, P, H)
            x_dense = x_dense[s]

            if self.similarity == 'dot':
                # (B, P, H) * (P, H) -> (B, P, H) -> (B, P) -> (B)
                sim_scores = x_dense * y_dense
                sim_scores = torch.sum(sim_scores, dim=1)
                sim_scores = torch.mean(sim_scores, dim=1)
                max_idx = torch.argmax(sim_scores, dim=0)
                max_idx = s[max_idx]
                x_patch = x_input_ids[max_idx]
                x_patch = self.tokenizer.decode(x_patch)

        y_split = y.split()
        patch_idx = list(range(len(y_split)))[idx]
        
        mixed = [] 
        for i, y_token in enumerate(y_split):
            if i in patch_idx:
                if i == patch_idx[0]:
                    mixed.append(x_patch.strip())
            else:
                mixed.append(y_token)
        mixed = ' '.join(mixed)

        return mixed

    def _make_mixup_indices(self, samples, rates):
        indices_list = []
        for sample in samples:
            indices = []
            sample = sample.split()
            for rate in rates:
                n_lost = round(len(sample) * rate)

                if len(sample) - n_lost <= 0:
                    idx = slice(None, None)
                else:
                    start = np.random.randint(0, len(sample) - n_lost + 1)
                    end = start + n_lost
                    idx = slice(start, end)

                if len(sample[idx]) == 0:
                    idx = np.random.randint(0, len(sample))
                    idx = slice(idx, idx + 1)

                indices.append(idx)
            indices_list.append(indices)
        return indices_list

    def __str__(self):
        return 'cut'