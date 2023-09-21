from abc import (
    ABC,
    abstractmethod,
)

from ..mixup_operators import BaseMixupOperator


class OODScoreCalculator(ABC):
    utilize_mixup = True

    @abstractmethod
    def load_model(self, backbone_name, device):
        pass

    def on_eval_start(
        self, 
        seen_labels, 
        given_images, 
        mixup_fn: BaseMixupOperator,
        ref_images,
        rates,
    ):
        pass

    def on_eval_end(self):
        pass
    
    @abstractmethod
    def process_images(
        self,
        images,
    ):
        pass
    
    @abstractmethod
    def select_given_images(
        self,
        given_images,
        **kwargs,
    ):
        pass
    
    # TODO: Remove this after integrating text, image code.
    def process_mixup_images(
        self,
        knwon_images,
        input_ids,
    ):
        pass

    @abstractmethod
    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        pass
    
    @abstractmethod
    def calculate_base_scores(
        self,
        **kwargs,
    ):
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass

    @property
    def ref_mode(self) -> str:
        return self._ref_mode

    @ref_mode.setter
    def ref_mode(self, ref_mode: str) -> str:
        self._ref_mode = ref_mode
    
    def transform(self, images):
        return images

    def post_transform(self, images):
        return images