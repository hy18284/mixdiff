from abc import (
    ABC,
    abstractmethod,
)


class OODScoreCalculator(ABC):
    utilize_mixup = True

    @abstractmethod
    def load_model(self, backbone_name, device):
        pass

    def on_eval_start(self, seen_labels):
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
    
    @abstractmethod
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

