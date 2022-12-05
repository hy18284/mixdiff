from abc import ABC, abstractmethod


class BaseOODDataModule(ABC):
    @abstractmethod
    def get_splits(self, n_samples_per_class: int, seed: int):
        pass

    @abstractmethod
    def construct_loader(self, batch_size: int):
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass