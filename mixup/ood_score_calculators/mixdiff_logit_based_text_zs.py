import itertools
from typing import (
    Optional,
    List,
)

import torch
import wandb
from transformers import (
    AutoModelForPreTraining,
    AutoTokenizer,
    BertForNextSentencePrediction,
)
from tqdm import tqdm

from .ood_score_calculator import OODScoreCalculator
from ..mixup_operators import BaseMixupOperator
from ..utils import log_mixup_samples


class MixDiffLogitBasedMixinTextZS:
    def __init__(
        self,
        batch_size: int,
        utilize_mixup: bool = True,
        add_base_score: bool = True,
        selection_mode: str = 'argmax',
        reverse: bool = True,
        template: str = 'I want to {}.',
        ignore_model_path: bool = True,
    ):
        self.batch_size = batch_size
        self.utilize_mixup = utilize_mixup
        self.add_base_score = add_base_score
        self.selection_mode = selection_mode
        self.known_mixup_table = wandb.Table(['x', 'y', 'mixup', 'rate', 'split'])
        self.reverse = reverse
        self.template = template
        self.ignore_model_path = ignore_model_path
    
    def load_model(self, backbone_name, device):
        self.device = device
        self.model = BertForNextSentencePrediction.from_pretrained(backbone_name)
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
        self.class_names = [
            self.template.format(seen_label)
            for seen_label in seen_labels
        ]

        if model_path is not None and not self.ignore_model_path:
            self.model = BertForNextSentencePrediction.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()

        if not self.utilize_mixup:
            return

        if self.selection_mode == 'euclidean' or self.selection_mode == 'dot':
            ValueError('Invalid selection option.')
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
                # (M * P * R) -> (M * P * R, NC, 2)
                logits = self._process_samples(mixed)
                self.oracle_logits.append(logits)

            # [NC, (M * P * R, NC, 2)] -> (NC, M * P * R, NC, 2)
            self.oracle_logits = torch.stack(self.oracle_logits, dim=0)
             
    def on_eval_end(self, iter_idx: int):
        self.id_logits = None
        self.oracle_logits = None
        self.mixup_states = None
        self.class_names = None
        
        wandb.log({
            f'oracle_mixup': self.known_mixup_table,
        })

    @torch.no_grad()
    def process_images(
        self,
        images,
    ):
        # (N) -> (N, NC, 2)
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
            # (N, NC, 2) -> (N, NC)
            probs = torch.softmax(logits)[:, :, 0]
            max_indices = torch.argmax(probs, dim=-1)
            # (NC, M * P * R, NC, 2) -> (N, M * P * R, NC, 2)
            chosen_images = [given_images[idx] for idx in max_indices]
            return chosen_images

        elif self.selection_mode == 'euclidean' or self.selection_mode == 'dot':
            ValueError('Invalid selection option.')
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
        # (N * R) -> (N * R, NC, 2) 
        return self._process_samples(images)

    @torch.no_grad()
    def process_mixed_oracle(
        self,
        images,
        logits,
        **kwargs
    ):
        if self.ref_mode == 'oracle' or self.ref_mode == 'rand_id':
            # (N, NC, 2) -> (N, NC)
            probs = torch.softmax(logits)[:, :, 0]
            max_indices = torch.argmax(probs, dim=-1)
            # (NC, M, M, R, NC, 2) -> (N, M, M, R, NC, 2)
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
        # (B * NC)
        images_pair = []
        for image in images:
            for class_name in self.class_names:
                if self.reverse:
                    images_pair.append((image, class_name))
                else:
                    images_pair.append((class_name, image))
        images = images_pair

        images_list = [
            images[i : i + self.batch_size]
            for i in range(0, len(images), self.batch_size)
        ]

        logits_list = []
        for pairs in images_list:
            images, class_names = list(zip(*pairs))
            batch = self.tokenizer(
                images,
                class_names,
                add_special_tokens=True,
                padding=True,
                return_tensors='pt',
                return_attention_mask=True,
            )
            for key in batch:
                batch[key] = batch[key].to(self.device)
            split_logits = self.model(**batch).logits
            logits_list.append(split_logits) 

        # (B, NC, 2)
        return torch.cat(logits_list, dim=0)