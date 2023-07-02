from typing import (
    Optional,
)

import torch
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
import clip
from tqdm import tqdm

from .ood_score_calculator import OODScoreCalculator
from ..mixup_operators.base_mixup_operator import BaseMixupOperator
from ..utils import log_mixup_samples


class MixDiffLogitBasedMixin:
    def __init__(
        self,
        batch_size: int,
        utilize_mixup: bool = True,
        add_base_score: bool = True,
    ):
        self.batch_size = batch_size
        self.utilize_mixup = utilize_mixup
        self.add_base_score = add_base_score
    
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

    
    def on_eval_end(self, iter_idx: int):
        del self.prompts_embeds
        del self.seen_labels

    @torch.no_grad()
    def process_images(
        self,
        images,
    ):
        image_embeds = self.clip_model.encode_image(images)
        image_embeds /= torch.norm(image_embeds, dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_embeds @ self.prompts_embeds.t()
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
        # (N)
        max_indices = torch.argmax(logits, dim=-1)
        # (NC, M, C, H, W) -> (N, M, C, H, W)
        chosen_images = given_images[max_indices, ...]
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
        **kwrags,
    ):
        return self._process_mixup_images(images)
    

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
        return torch.cat(logits_list, dim=0)

    def __str__(self) -> str:
        if not self.utilize_mixup:
            return f'{self.name}'
        if self.add_base_score:
            return f'mixdiff_{self.name}+'
        else:
            return f'mixdiff_{self.name}'