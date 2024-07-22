from typing import (
    Optional,
)

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .base_backbone import BaseBackbone
from ...mixup_operators.base_mixup_operator import BaseMixupOperator


class ClipBackbone(BaseBackbone):
    def load_model(self, backbone_name, device):
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(backbone_name)
        self.model.to(device)
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

    def process_images(self, images):
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
