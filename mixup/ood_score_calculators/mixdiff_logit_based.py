from typing import (
    Optional,
)

import torch
import torch.nn.functional as F
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
import clip
from tqdm import tqdm
import wandb

from .ood_score_calculator import OODScoreCalculator
from ..mixup_operators.base_mixup_operator import BaseMixupOperator
from ..utils import log_mixup_samples


class MixDiffLogitBasedMixin:
    def __init__(
        self,
        batch_size: int,
        utilize_mixup: bool = True,
        add_base_score: bool = True,
        selection_mode: str = 'argmax',
        intermediate_state: str = 'logit',
        oracle_sim_mode: str = 'uniform',
        oracle_sim_temp: float = 1.0,
        log_interval: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.utilize_mixup = utilize_mixup
        self.add_base_score = add_base_score
        self.selection_mode = selection_mode
        self.intermediate_state = intermediate_state
        self.oracle_sim_temp = oracle_sim_temp
        self.oracle_sim_mode = oracle_sim_mode

        self.log_interval = log_interval
        self.known_mixup_table = wandb.Table(['x', 'y', 'mixup', 'rate', 'split'])

        assert self.intermediate_state in ('logit', 'softmax')
        assert self.selection_mode in ('argmax', 'dot', 'euclidean')
        assert self.oracle_sim_mode in ('uniform', 'l2', 'dot', 'cosine_sim')
    
    def load_model(self, backbone_name, device):
        self.clip_model, _ = clip.load(
            backbone_name, 
            device=device, 
            download_root='trained_models'
        )
        self.clip_model.eval()
        self.cliptokenizer = clip_tokenizer()
        self.device = device

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
        given_images = given_images.to(self.device)
        if ref_images is not None:
            ref_images = ref_images.to(self.device)

        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]
        prompts_ids = [clip.tokenize(prompt) for prompt in seen_descriptions]
        prompts_ids = torch.cat(prompts_ids, dim=0).to(self.device)
        prompts_ids = prompts_ids.to(self.device)
        self.prompts_embeds = self.clip_model.encode_text(prompts_ids)
        self.prompts_embeds /= torch.norm(self.prompts_embeds, dim=-1, keepdim=True)
        self.seen_labels = seen_labels

        if self.selection_mode in ('euclidean', 'dot') or self.oracle_sim_mode != 'uniform':
            NC, M, C, H, W = given_images.size()
            given_images_flat = given_images.view(-1, C, H, W)
            self.id_logits = self._process_mixup_images(given_images_flat)

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
                    oracle=given, 
                    references=ref, 
                    rates=rates,
                    seed=seed,
                )

                if self.log_interval is not None and i % self.log_interval == 0:
                    log_mixup_samples(
                        ref_images=[ref],
                        known_mixup_table=self.known_mixup_table, 
                        known_mixup=mixed,
                        chosen_images=given.unsqueeze(0),
                        rates=rates, 
                        j=iter_idx,
                        N=1,
                        P=len(ref),
                        R=len(rates),
                        M=len(given)
                    )
                # (M * P * R, C, H, W) -> (M * P * R, NC)
                logits = self._process_mixup_images(mixed)
                self.oracle_logits.append(logits)
            # [NC, (M * M * R, NC)] -> (NC, M * M * R, NC)
            self.oracle_logits = torch.stack(self.oracle_logits, dim=0)

        self.NC = given_images.size(0)
        self.M = len(given_images[0])
        if self.ref_mode == 'rand_id':
            self.P = len(ref_images)
        elif self.ref_mode == 'oracle':
            self.P = self.M
        self.R = len(rates)

    def on_eval_end(self, iter_idx: int):
        del self.prompts_embeds
        del self.seen_labels

        self.id_logits = None
        self.oracle_logits = None
        self.sim_oracle = None

        wandb.log({
            'oracle_mixup': self.known_mixup_table,
        })

    @torch.no_grad()
    def process_images(
        self,
        images,
    ):
        image_embeds = self.clip_model.encode_image(images)
        image_embeds /= torch.norm(image_embeds, dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_embeds @ self.prompts_embeds.t()

        if self.intermediate_state == 'softmax':
            logits = torch.softmax(logits, dim=-1)

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
            # (N)
            max_indices = torch.argmax(logits, dim=-1)
            # (NC, M, C, H, W) -> (N, M, C, H, W)
            chosen_images = given_images[max_indices, ...]
        # elif self.selection_mode == 'euclidean' or self.selection_mode == 'dot':
        #     # (N, NC), (NC * M, NC) -> (N, NC * M)
        #     if self.selection_mode == 'euclidean':
        #         dists = torch.cdist(logits, self.id_logits, p=2)
        #     if self.selection_mode == 'dot':
        #         dists = -(logits @ self.id_logits.t())
        #     # (N, NC * M) -> (N, M)
        #     _, topk_indices = torch.topk(
        #         dists, 
        #         dim=1, 
        #         k=len(given_images[0]),
        #         largest=False,
        #         sorted=True,
        #     )
        #     # (N, M), (NC, M) -> (N, M)

        #     # (NC, M, C, H, W) -> (NC * M, C, H, W)
        #     given_images = torch.flatten(given_images, 0, 1)
        #     given_images = torch.gather(given_images, 1, topk_indices)
        # else:
        #     ValueError('Invalid selection option.')

        if self.oracle_sim_mode != 'uniform':
            # (NC, M, NC) -> (N, M, NC)
            id_logits = self.id_logits.view(self.NC, self.M, self.NC)
            self.sim_oracle = id_logits[max_indices, ...]

        return chosen_images
    
    @torch.no_grad()
    def process_mixed_target(
        self,
        images,
        **kwrags,
    ):
        return self._process_mixup_images(images)
    
    @torch.no_grad()
    def process_mixed_oracle(
        self,
        images,
        logits,
        **kwrags,
    ):
        if self.ref_mode == 'oracle' or self.ref_mode == 'rand_id':
            # (N, NC) -> (N)
            max_indices = torch.argmax(logits, dim=-1)
            # (NC, M * P * R, NC) -> (N, M * P * R, NC)
            mixed_logits = self.oracle_logits[max_indices]
        elif self.ref_mode == 'in_batch':
            mixed_logits = self._process_mixup_images(images)
        else:
            raise ValueError(f'Invalid ref_mode: {self.ref_mode}')

        N, NC = logits.size()
        if self.oracle_sim_mode == 'uniform':
            return mixed_logits

        if self.ref_mode not in ('oracle', 'rand_id'):
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
        
        # Make sure that proper average can be taken.
        if self.ref_mode == 'oracle':
            mixed_logits = mixed_logits * (self.M - 1)
        else:
            mixed_logits = mixed_logits * self.M

        return mixed_logits

    @torch.no_grad()
    def _process_mixup_images(
        self,
        images,
    ):
        images_list = torch.split(images, self.batch_size, dim=0)
        logits_list = []
        for split_images in images_list:
            image_embeds = self.clip_model.encode_image(split_images)
            image_embeds /= torch.norm(image_embeds, dim=-1, keepdim=True)
            logit_scale = self.clip_model.logit_scale.exp()
            split_logits = logit_scale * image_embeds @ self.prompts_embeds.t()
            logits_list.append(split_logits) 
        
        logits = torch.cat(logits_list, dim=0)
        if self.intermediate_state == 'softmax':
            logits = torch.softmax(logits, dim=-1)
        return logits

    def __str__(self) -> str:
        if self.intermediate_state == 'logit':
            inter_state = ''
        elif self.intermediate_state == 'softmax':
            inter_state = '_s'

        if not self.utilize_mixup:
            return f'{self.name}'
        if self.add_base_score:
            return f'mixdiff_{self.name}{inter_state}+'
        else:
            return f'mixdiff_{self.name}{inter_state}'