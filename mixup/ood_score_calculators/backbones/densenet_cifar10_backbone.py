from typing import (
    Optional,
)

import torch

from .base_backbone import BaseBackbone
from ...mixup_operators.base_mixup_operator import BaseMixupOperator
from .densenet import DenseNet3


class DenseNetCIFAR10Backbone(BaseBackbone):
    def load_model(self, backbone_name, device):
        self.model = DenseNet3(100, 10)
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