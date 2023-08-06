from typing import (
    Optional,
)

import torch

from .base_backbone import BaseBackbone
import clip
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
from ...mixup_operators.base_mixup_operator import BaseMixupOperator
from .resnet_imagenet import ResNet50


class ResNetBackbone(BaseBackbone):
    def load_model(self, backbone_name, device):
        self.model = ResNet50()
        self.model.load_state_dict(torch.load(backbone_name))
        self.model.to(device)
        self.model.eval()

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
        pass

    def on_eval_end(self, iter_idx: int):
        pass

    def process_images(self, images):
        return self.model(images)