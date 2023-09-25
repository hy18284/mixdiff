from typing import (
    Optional,
)

import torch
from torchvision import transforms

from .base_backbone import BaseBackbone
import clip
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
from ...mixup_operators.base_mixup_operator import BaseMixupOperator
from .resnet_imagenet import ResNet50


class ResNetBackbone(BaseBackbone):
    def __init__(self, post_transform: bool = False) -> None:
        super().__init__()
        if post_transform:
            self.transform_fn = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
            self.post_transform_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        else:
            self.transform_fn = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.post_transform_fn = lambda x: x

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

    def transform(self, images):
        return self.transform_fn(images)
   
    def post_transform(self, images):
        orig_size = images.size()
        C, H, W = orig_size[-3:]
        images = images.view(-1, C, H, W)
        images = self.post_transform_fn(images)
        images = images.view(orig_size)
        return images