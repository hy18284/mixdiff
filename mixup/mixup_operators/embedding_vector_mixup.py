import itertools
from typing import (
    List,
    Tuple,
    Union,
    Optional,
)

from transformers import (
    AutoModel,
    AutoTokenizer,
)
import torch
from torch.nn import Embedding
import numpy as np

from .base_mixup_operator import BaseMixupOperator


class EmbeddingVectorMixup(BaseMixupOperator):
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
        self.embedding: Embedding = model.get_input_embeddings()
        self.embedding.to(self.device)

        self.token_type_embeddings = model.embeddings.token_type_embeddings
        self.token_type_embeddings.to(device)

        pad_id = self.tokenizer.pad_token_id
        self.embedding.weight[pad_id] = 0.0

    def __call__(
        self, 
        references: List[str],
        rates: List[float],
        oracle: List[List[str]] = None,
        targets: List[str] = None, 
        seed: Optional[int] = None,
    ) -> Tuple[List[List[List[List[str]]]], List[List[List[str]]]]:
        """Performs mixup and returns oracle, in-batch mixed samples

        Args:
            oracle (List[List[str]]): List of strings with size (N, M).
            targets (List[str]): List of strings with size (N).
            references (List[str]): List of strings with size (P).
            rates (List[str]): List of mixup rates with size (R).

        Returns:
            Tuple[List[List[List[List[str]]]], List[List[List[str]]]]:
            Mixed oracle, in-batch samples each of which has size of
            (N, M, N, R), (N, N, R), respectively.
        """
        P = len(references)
        R = len(rates)
         
        if oracle is not None:
            N = len(oracle)
            M = len(oracle[0])
            
            oracle = itertools.chain.from_iterable(oracle)
            batch = oracle + references
            batch = self.tokenizer(
                batch,
                add_special_tokens=True,
                padding=True,
                return_tensors='pt',
                return_attention_mask=True,
            )
            oracle_ids = batch['input_ids'][:-P].to(self.device)
            ref_ids = batch['input_ids'][-P:].to(self.device)

            # (N * M + P, L, H)
            embeds = self.embedding(batch['input_ids'])
            L = embeds.size(1)

            # (N, M, P, R, L, H)
            mixed = self._oracle_mixup(embeds, rates, N, M, P)
            if 'token_type_ids' in batch:
                type_embeds = self.token_type_embedding(batch['token_type_ids'])
                type_mixed = self._oracle_mixup(type_embeds, rates, N, M, P)
            
            # (N * M + P, L) -> (N, M, L)
            orcl_attn_mask = batch['attention_mask'][:-P].view(N, M, -1)
            # (P, L)
            ref_attn_mask = batch['attention_mask'][-P:].unsqueeze(0)

            # (N, M, 1, L) (1, 1, P, L) -> (N, M, P, L)
            attn_mask = torch.logical_or(orcl_attn_mask, ref_attn_mask)
            # (N, M, P, 1, L) -> (N, M, P, R, L)
            attn_mask = attn_mask.unsqueeze(-2)
            attn_mask = attn_mask.expand(-1, -1, -1, -1, R, -1)
            # (N, M, P, L) -> (N, M, P, R, L)
            attn_mask = attn_mask.view(N * M * P * R, -1)
            attn_mask = attn_mask.to(torch.long)

            oracle_mixup = {
                'inputs_embeds': mixed,
                'attention_mask': attn_mask,
            }

            if targets is None:
                return oracle_mixup
            
        if targets is not None:
            N = len(targets)

            # (N * P * R) 
            batch = targets + references
            batch = self.tokenizer(
                batch,
                add_special_tokens=True,
                padding=True,
                return_tensors='pt',
                return_attention_mask=True,
            )

            # (N + P, L, H)
            embeds = self.embedding(batch['input_ids'])
            L = embeds.size(1)
            mixed = self._target_mixup(embeds, rates, N, P)

            # (N + P, L) -> (N, L)
            orcl_attn_mask = batch['attention_mask'][:-P].unsqueeze(1)
            # (P, L)
            ref_attn_mask = batch['attention_mask'][-P:].unsqueeze(0)

            # (N, 1, L) (1, P, L) -> (N, P, L)
            attn_mask = torch.logical_or(orcl_attn_mask, ref_attn_mask)
            # (N, P, L) -> (N, P, R, L)
            attn_mask = attn_mask.unsqueeze(2).expand(-1, -1, R, -1)
            # (N, M, P, L) -> (N, M, P, R, L)
            attn_mask = attn_mask.view(N * M * P * R, -1)
            attn_mask = attn_mask.to(torch.long)

            target_mixup = {
                'inputs_embeds': mixed,
                'attention_mask': attn_mask,
            }
            if oracle is None:
                return target_mixup
            
        return oracle_mixup, target_mixup
    
    def _oracle_mixup(
        self, 
        embeds,
        rates: torch.Tensor, 
        N, 
        M,
        P,
    ):
        L = embeds.size(1)

        # (N, M, L, H) 
        oracle_embeds = embeds[:-P, ...].view(N, M, L, -1)
        # (P, L, H) 
        ref_embeds = embeds[-P:, ...]

        # (P, 1, L, H) * (R) -> (P, R, L, H) 
        ref_embeds = ref_embeds.unsqueeze(1) * (1 - rates)

        # (N, M, 1, L, H) * (R) -> (N, M, R, L, H)
        oracle_embeds = oracle_embeds.unsqueeze(2) * rates

        # (1, 1, P, R, L, H) + (N, M, 1, R, L, H) -> (N, M, P, R, L, H)
        mixed = ref_embeds.unsqueeze[None, None, ...] + oracle_embeds.unsqueeze(2)
        return mixed
    
    def _target_mixup(
        self,
        embeds,
        rates: torch.Tensor, 
        N, 
        P,
    ):
        L = embeds.size(1)

        # (N, L, H) 
        target_embeds = embeds[:-P, ...].view(N, L, -1)

        # (P, L, H) 
        ref_embeds = embeds[-P:, ...].to

        # (P, 1, L, H) * (R) -> (P, R, L, H) 
        ref_embeds = ref_embeds.unsqueeze(1) * (1 - rates)

        # (N, 1, L, H) * (R) -> (N, R, L, H)
        target_embeds = target_embeds.unsqueeze(1) * rates

        # (1, P, R, L, H) + (N, 1, R, L, H) -> (N, P, R, L, H)
        mixed = ref_embeds.unsqueeze(0) + target_embeds.unsqueeze(1)
        return mixed

    def __str__(self):
        return 'ebv'