import itertools
from typing import (
    Optional,
    List,
)

import torch
import torch.nn.functional as F
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
        intermediate_state: str = 'logit',
        oracle_sim_mode: str = 'uniform',
        oracle_sim_temp: float = 1.0,
        aux_sim_mode: str = 'uniform',
        aux_sim_temp: float = 1.0,
    ):
        self.batch_size = batch_size
        self.utilize_mixup = utilize_mixup
        self.add_base_score = add_base_score
        self.selection_mode = selection_mode
        self.intermediate_state = intermediate_state
        self.oracle_sim_mode = oracle_sim_mode
        self.oracle_sim_temp = oracle_sim_temp
        self.aux_sim_mode = aux_sim_mode
        self.aux_sim_temp = aux_sim_temp
        assert self.selection_mode in ('argmax', 'dot', 'euclidean')
        assert self.intermediate_state in ('logit', 'softmax')
        assert self.oracle_sim_mode in ('uniform', 'l2', 'dot', 'cosine_sim')
        assert self.aux_sim_mode in ('uniform', 'l2')
        self.known_mixup_table = wandb.Table(['x', 'y', 'mixup', 'rate', 'split'])
    
    def load_model(self, backbone_name, device):
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(backbone_name)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)

    def on_eval_start(
        self, 
        seen_labels, 
        given_images, 
        mixup_fn: Optional[BaseMixupOperator],
        ref_images,
        rates,
        seed,
        iter_idx,
        model_path,
    ):
        if model_path is not None:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()

        if not self.utilize_mixup:
            return
        
        if self.selection_mode == 'euclidean' or self.selection_mode == 'dot' \
            or self.oracle_sim_mode != 'uniform':
            given_images_flat = list(itertools.chain.from_iterable(given_images))
            images_list = [
                given_images_flat[i : i + self.batch_size]
                for i in range(0, len(given_images_flat), self.batch_size)
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
                if self.intermediate_state == 'softmax':
                    split_logits = torch.softmax(split_logits, dim=-1)
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
            # [NC, (M * M * R, NC)] -> (NC, M * M * R, NC)
            self.oracle_logits = torch.stack(self.oracle_logits, dim=0)

        self.M = len(given_images[0])
        if self.ref_mode == 'rand_id':
            self.P = len(ref_images)
        elif self.ref_mode == 'oracle':
            self.P = self.M
        self.R = len(rates)
             
    def on_eval_end(self, iter_idx: int):
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
            chosen_images = chosen_images_list
        else:
            ValueError('Invalid selection option.')
        
        if self.oracle_sim_mode != 'uniform':
            self.sim_oracle = []
            for chosen in chosen_images:
                sim_oracle = self._process_samples(chosen)
                self.sim_oracle.append(sim_oracle)
            # (N, M, NC)
            self.sim_oracle = torch.stack(self.sim_oracle, dim=0)

        return chosen_images

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
            # (NC, M * P * R, NC) -> (N, M * P * R, NC)
            mixed_logits = self.oracle_logits[max_indices]
        elif self.ref_mode == 'in_batch':
            mixed_logits = self._process_samples(images)
        else:
            raise ValueError(f'Invalid ref_mode: {self.ref_mode}')
        
        N, NC = logits.size()
        if self.oracle_sim_mode == 'uniform':
            return mixed_logits

        if self.ref_mode != 'oracle':
            raise ValueError('Not yet implemented for other ref_modes')

        if self.ref_mode == 'in_batch':
            self.P = len(logits)
        
        if self.oracle_sim_mode == 'l2':
            # (N, 1, NC), (N, M, NC) -> (N, 1, M) -> (N, M)
            sim_scores = -torch.cdist(logits.unsqueeze(1), self.sim_oracle, p=2)
            sim_scores = sim_scores.squeeze()
        elif self.oracle_sim_mode == 'dot':
            # (N, 1, NC), (N, M, NC) -> (N, M, NC) -> (N, M)
            sim_scores = logits.unsqueeze(1) * self.sim_oracle
            sim_scores = sim_scores.sum(dim=-1)
        elif self.oracle_sim_mode == 'cosine_sim':
            # (N, 1, NC), (N, M, NC) -> (N, M)
            sim_scores = F.cosine_similarity(
                logits.unsqueeze(1), 
                self.sim_oracle, 
                dim=-1,
            )

        # (N,  M,  P,  R, NC)
        mixed_logits = mixed_logits.view(N, self.M, self.P, self.R, NC)

        # (N, M, 1) -> (N, M, P)
        sim_scores = sim_scores.unsqueeze(-1).repeat(1, 1, self.P)
        if self.ref_mode == 'oracle':
            diags = torch.diagonal(sim_scores, dim1=1, dim2=2)
            diags.fill_(torch.tensor(float('-inf'), device=self.device))
        sim_scores = torch.softmax(sim_scores / self.oracle_sim_temp, dim=1)

        # (N, M, P, R, NC) * (N, M, P, 1, 1) -> (N, M, P, R, NC)
        mixed_logits = mixed_logits * sim_scores[:, :, :, None, None]
        mixed_logits = mixed_logits * (self.M - 1)

        return mixed_logits
    
    def process_mixup_images(self, images):
        pass

    def __str__(self) -> str:
        if self.selection_mode == 'euclidean':
            sel_mode = 'eucl'
        elif self.selection_mode == 'argmax':
            sel_mode = 'agmax'
        elif self.selection_mode == 'dot':
            sel_mode = 'dot'
        
        if self.intermediate_state == 'logit':
            inter_state = ''
        elif self.intermediate_state == 'softmax':
            inter_state = '_s'

        if not self.utilize_mixup:
            return f'{self.name}'
        if self.add_base_score:
            return f'mixdiff_{self.name}_{sel_mode}{inter_state}+'
        else:
            return f'mixdiff_{self.name}_{sel_mode}{inter_state}'
    
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
            if self.intermediate_state == 'softmax':
                split_logits = torch.softmax(split_logits, dim=-1)
            logits_list.append(split_logits) 
        return torch.cat(logits_list, dim=0)