from abc import ABC, ABCMeta, abstractmethod
from typing import (
    List,
    Tuple,
    Optional,
    Any,
)


class BaseMixupOperator(ABC):
    @abstractmethod
    def __call__(
        self, 
        oracle: List[List[str]], 
        references: List[str],
        targets: List[str], 
        rates: List[str],
        use_registered_state: bool=False,
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

    def unregister_state(self):
        pass
    
    def get_state(
        self,
        oracle: Optional[List[List[str]]] = None, 
        references: Optional[List[str]] = None,
        targets: Optional[List[str]] = None, 
        rates: Optional[List[str]] = None,
        state: Optional[Any] = None
    ):
        pass
    