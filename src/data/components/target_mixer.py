from __future__ import annotations

from abc import abstractmethod, ABC

import numpy as np


class BaseTargetMixer(ABC):
    @abstractmethod
    def mix_targets(self, targets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Return boundary based on features
        Args:
            targets (dict[str, np.ndarray]): target generated in previous steps
        Returns:
            list[int], each element of list is position of interval end (day, month etc)
        """


class TargetMixerCompose:
    def __init__(self, target_mixers: list[BaseTargetMixer]):
        self.target_mixers = target_mixers

    def __len__(self):
        return len(self.target_mixers)

    def mix_targets(self, targets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for target_mixer in self.target_mixers:
            target_mixer.mix_targets(targets)
        return targets


class LinearBaseTargetMixer(BaseTargetMixer):
    def __init__(self,
                 target_names: list[str],
                 target_coefficients: list[float],
                 output_target_names: str
                 ):
        super().__init__()
        self.target_names = target_names
        self.target_coefficients = target_coefficients
        self.output_target_names = output_target_names
        assert len(self.target_names) == len(self.target_coefficients), "length of features " \
                                                                        "and coefficients should be equals"

    def mix_targets(self, targets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        total = targets[self.target_names[0]] * self.target_coefficients[0]

        for idx in range(1, len(self.target_names)):
            coefficient = self.target_coefficients[idx]
            target = targets[self.target_names[idx]]
            total += coefficient * target

        targets[self.output_target_names] = total

        return targets


class DividerBaseTargetMixer(BaseTargetMixer):
    def __init__(self,
                 dividend_name: str,
                 divisor_name: str,
                 output_target_names: str
                 ):
        super().__init__()
        self.dividend_name = dividend_name
        self.divisor_name = divisor_name
        self.output_target_names = output_target_names

    def mix_targets(self, targets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        dividend = targets[self.dividend_name]
        divisor = targets[self.divisor_name]
        total = dividend / divisor
        targets[self.output_target_names] = total
        return targets
