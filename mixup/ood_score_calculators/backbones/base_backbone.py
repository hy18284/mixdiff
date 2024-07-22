from typing import (
    Optional,
)
from abc import (
    ABC,
    abstractmethod,
)

from ...mixup_operators.base_mixup_operator import BaseMixupOperator


class BaseBackbone(ABC):
    def load_model(self, backbone_name, device):
        pass

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
        few_shot_samples,
    ):
        pass

    def on_eval_end(self, iter_idx: int):
        pass

    @abstractmethod
    def process_images(self, images):
        pass

    def transform(self, images):
        return images

    def post_transform(self, images):
        return images