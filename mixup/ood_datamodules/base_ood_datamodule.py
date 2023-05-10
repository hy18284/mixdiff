from abc import ABC, abstractmethod


class BaseOODDataModule(ABC):
    def __init__(self, ref_mode: str='in_batch'):
        self._ref_mode = ref_mode

    @abstractmethod
    def get_splits(self, n_samples_per_class: int, seed: int):
        pass

    @abstractmethod
    def construct_loader(self, batch_size: int, shuffle: bool = True):
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @property
    def ref_mode(self) -> str:
        self._ref_mode

    @ref_mode.setter
    def ref_mode(self, ref_mode: str) -> str:
        self._ref_mode = ref_mode
        