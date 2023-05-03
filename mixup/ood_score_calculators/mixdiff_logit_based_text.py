import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .ood_score_calculator import OODScoreCalculator


class MixDiffLogitBasedMixinText:
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
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(backbone_name)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)

    def on_eval_start(self, seen_labels):
        pass
    
    def on_eval_end(self):
        pass

    @torch.no_grad()
    def process_images(
        self,
        images,
    ):
        batch = self.tokenizer(
            images, 
            add_special_tokens=True,
            padding=True,
            return_tensors='pt',
            return_attention_mask=True,
        )
        for key in batch:
            batch[key] = batch[key].to(self.device)

        logits = self.model(**batch).logits

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
        # (N, NC) -> (N)
        max_indices = torch.argmax(logits, dim=-1)
        chosen_images = [given_images[idx] for idx in max_indices]
        return chosen_images

    @torch.no_grad()
    def process_mixup_images(
        self,
        images,
    ):
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

    def __str__(self) -> str:
        if not self.utilize_mixup:
            return f'{self.name}'
        if self.add_base_score:
            return f'mixdiff_{self.name}+'
        else:
            return f'mixdiff_{self.name}'