from typing import (
    List,
    Tuple,
    Optional,
)

from .base_mixup_operator import BaseMixupOperator


class ConcatMixup(BaseMixupOperator):
    def __init__(self, ref_pos: str):
        self.ref_pos = ref_pos
        assert self.ref_pos in ['front', 'rear', 'both']
        
    def __call__(
        self, 
        references: List[str],
        rates: List[float],
        oracle: List[List[str]] = None,
        targets: List[str] = None, 
        seed: Optional[int] = None,
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
        rates = rates.tolist()
        assert len(rates) < 4
        if len(rates) == 1:
            assert self.ref_pos in ['front', 'rear']
        elif len(rates) == 2:
            assert self.ref_pos in ['both']
        else:
            ValueError('Invalid rates')

        if oracle is not None:
            # (N * M * N * R) 
            oracle_mixup = []
            for oracle_images in oracle:
                for oracle_image in oracle_images:
                    for ref in references:
                        mixup = self.mixup(oracle_image, ref)
                        oracle_mixup += mixup

            if targets is None:
                return oracle_mixup
        
        if targets is not None:
            # (N * N * R) 
            target_mixup = [] 
            for target in targets:
                for ref in references:
                    mixup = self.mixup(target, ref) 
                    target_mixup += mixup

            if oracle is None:
                return target_mixup
            
        return oracle_mixup, target_mixup

    def mixup(self, x: str, ref: str):
        if self.ref_pos == 'front':
            return [' '.join([ref, x])]
        elif self.ref_pos == 'back':
            return [' '.join([x, ref])]
        else:
            return [
                ' '.join([ref, x]),
                ' '.join([x, ref])
            ]


    def __str__(self):
        if self.ref_pos == 'front':
            return 'caf'
        elif self.ref_pos == 'rear':
            return 'car'
        else:
            return 'cab'