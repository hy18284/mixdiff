from typing import Any
from abc import ABC, abstractmethod


class BaseOODDataModule(ABC):
    def __init__(self, ref_mode: str='in_batch'):
        self._ref_mode = ref_mode

    @abstractmethod
    def get_splits(
        self, 
        n_samples_per_class: int, 
        seed: int, 
        n_ref_samples: int,
        batch_size: int,
        shuffle: bool = True,
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
        
    @property
    def flatten(self) -> str:
        return False

    def post_transform(self, images) -> Any:
        return images