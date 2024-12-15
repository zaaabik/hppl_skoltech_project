from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from numba import njit


class FeatureAggregation(ABC):
    @abstractmethod
    def get_features_need_for_aggregate(self) -> list[str]:
        pass

    @abstractmethod
    def as_jit_function(self):
        pass

    @abstractmethod
    def __call__(self,
                 features: dict[str, np.ndarray]
                 ) -> float | int:
        pass


class FuncFeatureAggregation(FeatureAggregation):
    def __init__(self, field: str,
                 aggregate_function: Callable,
                 none_value: float = 0.):
        super().__init__()
        self.field = field
        self.aggregate_function = aggregate_function
        self.none_value = np.float32(none_value)

    def get_features_need_for_aggregate(self) -> list[str]:
        return [self.field]

    def as_jit_function(self):
        return apply_on_field_jit, (self.field, self.aggregate_function, self.none_value)

    def __call__(self, features: dict[str, np.ndarray]) -> float | int:
        feature = features[self.field]
        if len(feature) == 0:
            return self.none_value

        return self.aggregate_function(
            feature
        )


@njit(
    'float32(float32[:,:])', nogil=True
)
def sum_jit(array):
    return array.sum()


@njit(
    'float32(float32[:,:])', nogil=True
)
def mean_jit(array):
    return array.mean()


@njit(
    'float32(float32[:,:])', nogil=True
)
def max_jit(array):
    return array.max()


@njit(
    'float32(float32[:, :])', nogil=True
)
def min_jit(array):
    return array.min()


@njit(nogil=True)
def apply_on_field_jit(features: dict[str, np.ndarray],
                       field: str = 'amount_rur',
                       agg_function=sum_jit,
                       none_value: float = 0.) -> float:
    feature = features[field]
    if len(feature) == 0:
        return none_value
    result = agg_function(
        feature
    )
    return np.float32(result)
