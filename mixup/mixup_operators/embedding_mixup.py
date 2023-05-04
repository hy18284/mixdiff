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

from .base_mixup_operator import BaseMixupOperator


class EmbeddingMixup(BaseMixupOperator):
    def __init__(
        self, 
        interpolation: str,
        similarity: str,
        model_path: str, 
        device: Union[int, str],
    ):
        self.interploation = interpolation
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
        rates: List[str]
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

        # (N * M * N * R) 
        oracle_mixup = []
        for oracle_images in oracle:
            for oracle_image in oracle_images:
                for sample in samples:
                    mixup = [
                        self.mixup(oracle_image, sample, rate) 
                        for rate in rates
                    ]
                    oracle_mixup += mixup
        
        # (N * N * R) 
        target_mixup = [] 
        for sample in samples:
            for sample in samples:
                mixup = [
                    self.mixup(sample, sample, rate) 
                    for rate in rates
                ]
                target_mixup += mixup
        
        return oracle_mixup, target_mixup

    def mixup(self, x: str, y: str, rate: float):
        x_input_ids = self.tokenizer(
            x, 
            add_special_tokens=False, 
            padding=False, 
            return_tensors='pt',
        )['input_ids'].to(self.device)[0]
        y_input_ids = self.tokenizer(
            y, 
            add_special_tokens=False, 
            padding=False, 
            return_tensors='pt',
        )['input_ids'].to(self.device)[0]

        x_input_ids, y_input_ids = self._interpolate(x_input_ids, y_input_ids)

        mixed = rate * x_input_ids + (1 - rate) * y_input_ids
        if self.similarity == 'dot':
            # (L, H) * (H, V) -> (L, V)
            mixed = mixed @ self.embedding.weight.t()
            _, mixed = torch.max(mixed, dim=1)
        elif self.similarity == 'cosine':
            # (L, 1, H) * (V, H) -> (L, V)
            torch.cosine_similarity(mixed.unsqueeze(1), self.embedding.weight, dim=1)
            _, mixed = torch.max(mixed, dim=1)
            
        mixed = self.tokenizer.decode(mixed)

        return mixed
    
    def _interpolate(
        self, 
        x_input_ids: torch.Tensor, 
        y_input_ids: torch.tensor, 
    ):
        if self.interploation == 'nearest':
            if len(x_input_ids) > len(y_input_ids):
                idx = torch.linspace(
                    0, 
                    len(y_input_ids) - 1, 
                    len(x_input_ids), 
                    device=self.device
                )
                idx = torch.round(idx).to(torch.long)
                y_input_ids = y_input_ids[idx]
            elif len(y_input_ids) > len(x_input_ids):
                idx = torch.linspace(
                    0, 
                    len(x_input_ids) - 1, 
                    len(y_input_ids), 
                    device=self.device
                )
                idx = torch.round(idx).to(torch.long)
                x_input_ids = x_input_ids[idx]

        elif self.interploation == 'truncate':
            if len(x_input_ids) > len(y_input_ids):
                x_input_ids = x_input_ids[:len(y_input_ids)]
            elif len(y_input_ids) > len(x_input_ids):
                y_input_ids = y_input_ids[:len(x_input_ids)]
        
        x_input_ids = self.embedding(x_input_ids) 
        y_input_ids = self.embedding(y_input_ids) 

        if self.interploation == 'pad':
            if len(x_input_ids) > len(y_input_ids):
                y_paded = torch.zeros_like(x_input_ids)
                y_paded[:len(y_input_ids)] = y_input_ids
                y_input_ids = y_paded
            elif len(y_input_ids) > len(x_input_ids):
                x_paded = torch.zeros_like(y_input_ids)
                x_paded[:len(x_input_ids)] = x_input_ids
                x_input_ids = x_paded
        
        return x_input_ids, y_input_ids

    def __str__(self):
        return 'tkn_emb'