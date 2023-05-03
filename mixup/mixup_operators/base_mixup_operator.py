from abc import ABC, abstractmethod
from typing import (
    List,
    Tuple,
)


class BaseMixupOperator(ABC):
    @abstractmethod
    def __call__(
        self, 
        oracle: List[List[str]], 
        samples: List[str], 
        rates: List[str]
    ) -> Tuple[List[List[List[List[str]]]], List[List[List[str]]]]:
        """Performs mixup and returns oracle, in-batch mixed samples

        Args:
            oracle (List[List[str]]): List of strings with size (N, M).
            samples (List[str]): List of strings with size (N).
            rates (List[str]): List of mixup rates with size (R).

        Returns:
            Tuple[List[List[List[List[str]]]], List[List[List[str]]]]:
            Mixed oracle, in-batch samples each of which has size of
            (N, M, N, R), (N, N, R), respectively.
        """
        pass

    @abstractmethod
    def __str__(self):
        pass