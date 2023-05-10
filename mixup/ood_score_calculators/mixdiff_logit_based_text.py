import itertools
from typing import (
    Optional,
    List,
)

import torch
import wandb
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from tqdm import tqdm

from .ood_score_calculator import OODScoreCalculator
from ..mixup_operators import BaseMixupOperator
from ..utils import log_mixup_samples


class MixDiffLogitBasedMixinText:
    def __init__(
        self,
        batch_size: int,
        utilize_mixup: bool = True,
        add_base_score: bool = True,
        selection_mode: str = 'argmax',
    ):
        self.batch_size = batch_size
        self.utilize_mixup = utilize_mixup
        self.add_base_score = add_base_score
        self.selection_mode = selection_mode
    
    def load_model(self, backbone_name, device):
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(backbone_name)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)

        self.known_mixup_table = wandb.Table(['x', 'y', 'mixup', 'rate', 'split'])

    def on_eval_start(
        self, 
        seen_labels, 
        given_images, 
        mixup_fn: Optional[BaseMixupOperator],
        ref_images,
        rates,
        seed,
        iter_idx,
    ):
        if self.selection_mode == 'euclidean' or self.selection_mode == 'dot':
            given_images = list(itertools.chain.from_iterable(given_images))
            images_list = [
                given_images[i : i + self.batch_size]
                for i in range(0, len(given_images), self.batch_size)
            ]
            logits_list = []
            for split_images in images_list:
                batch = self.tokenizer(
                    split_images,
                    add_special_tokens=True,
                    padding=True,
                    return_tensors='pt',
                    return_attention_mask=True,
                )
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                split_logits = self.model(**batch).logits
                logits_list.append(split_logits) 
            self.id_logits = torch.cat(logits_list, dim=0)
        
        self.oracle_logits = [] 
        if self.ref_mode == 'oracle' or self.ref_mode == 'rand_id':
            for i, given in tqdm(
                list(enumerate(given_images)),
                desc='Processing oracle samples'
            ):
                if self.ref_mode == 'oracle':
                    ref = given
                elif self.ref_mode == 'rand_id':
                    ref = ref_images

                mixed = mixup_fn(
                    oracle=[given], 
                    references=ref, 
                    rates=rates,
                    seed=seed,
                )

                log_mixup_samples(
                    ref_images=[ref],
                    known_mixup_table=self.known_mixup_table, 
                    known_mixup=mixed,
                    chosen_images=[given],
                    rates=rates, 
                    j=iter_idx,
                    N=1,
                    P=len(ref),
                    R=len(rates),
                    M=len(given)
                )
                # (M * M * R) -> (M * M * R, NC)
                logits = self._process_samples(mixed)
                self.oracle_logits.append(logits)
            self.oracle_logits = torch.stack(self.oracle_logits, dim=0)
             
    def on_eval_end(self):
        self.id_logits = None
        self.oracle_logits = None
        self.mixup_states = None
        
        wandb.log({
            'oracle_mixup': self.known_mixup_table,
        })

    @torch.no_grad()
    def process_images(
        self,
        images,
    ):
        logits = self._process_samples(images)

        kwargs = {
            'logits': logits,
        }
        return kwargs
    
    @torch.no_grad()
    def select_given_images(
        self,
        given_images,
        logits,
        **kwargs,
    ):
        if self.selection_mode == 'argmax':
            # (N, NC) -> (N)
            max_indices = torch.argmax(logits, dim=-1)
            chosen_images = [given_images[idx] for idx in max_indices]
            return chosen_images

        elif self.selection_mode == 'euclidean' or self.selection_mode == 'dot':
            if self.selection_mode == 'euclidean':
                # (N, NC), (NC * M, NC) -> (N, NC * M) -> (N, M)
                dists = torch.cdist(logits, self.id_logits, p=2)
            if self.selection_mode == 'dot':
                # (N, NC), (NC, NC * M) -> (N, NC * M) -> (N, M)
                dists = -(logits @ self.id_logits.t())
            _, topk_indices = torch.topk(
                dists, 
                dim=1, 
                k=len(given_images[0]),
                largest=False,
                sorted=True,
            )
            given_images = list(itertools.chain.from_iterable(given_images))
            chosen_images_list = []
            for top_k_idx in topk_indices:
                chosen_images = [
                    given_images[idx] for idx in top_k_idx
                ]
                chosen_images_list.append(chosen_images)
            return chosen_images_list
        else:
            ValueError('Invalid selection option.')

    @torch.no_grad()
    def process_mixed_target(
        self,
        images,
        **kwargs
    ):
        return self._process_samples(images)

    @torch.no_grad()
    def process_mixed_oracle(
        self,
        images,
        logits,
        **kwargs
    ):
        if self.ref_mode == 'oracle' or self.ref_mode == 'rand_id':
            # (N, NC) -> (N)
            max_indices = torch.argmax(logits, dim=-1)
            # (NC, M, M, NC) -> (N, M, M, NC)
            return self.oracle_logits[max_indices]
        elif self.ref_mode == 'in_batch':
            return self._process_samples(images)
        else:
            raise ValueError(f'Invalid ref_mode: {self.ref_mode}')
    
    def process_mixup_images(self, images):
        pass

    def __str__(self) -> str:
        if self.selection_mode == 'euclidean':
            sel_mode = 'eucl'
        elif self.selection_mode == 'argmax':
            sel_mode = 'agmax'
        elif self.selection_mode == 'dot':
            sel_mode = 'dot'

        if not self.utilize_mixup:
            return f'{self.name}'
        if self.add_base_score:
            return f'mixdiff_{self.name}_{sel_mode}+'
        else:
            return f'mixdiff_{self.name}_{sel_mode}'
    
    def _process_samples(self, images: List[str]):
        images_list = [
            images[i : i + self.batch_size]
            for i in range(0, len(images), self.batch_size)
        ]
        logits_list = []
        for split_images in images_list:
            batch = self.tokenizer(
                split_images,
                add_special_tokens=True,
                padding=True,
                return_tensors='pt',
                return_attention_mask=True,
            )
            for key in batch:
                batch[key] = batch[key].to(self.device)
            split_logits = self.model(**batch).logits
            logits_list.append(split_logits) 
        return torch.cat(logits_list, dim=0)