from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numba import njit


class TransactionFilter(ABC):
    @abstractmethod
    def get_features_need_for_filter(self) -> list[str]:
        pass

    @abstractmethod
    def as_jit_function(self):
        pass

    @abstractmethod
    def get_transaction_indexes(self, features: dict[str, np.ndarray]) -> list[int]:
        """
        Return boundary based on features
        Args:
            features (dict[str, np.ndarray]): dict of features each field is [L,d], L - length of seq,
                d - size of field (often 1, but for PLE could be greater)
        Returns:
            list[int], each element of list is position of interval end (day, month etc)
        """


@njit(
    'int32[:](DictType(unicode_type, float32[:, :]), unicode_type, int32)', nogil=True
)
def get_transaction_eq_field_indexes_jit(features, field, value):
    indexes = np.where(features[field].astype('int32') == value)
    return indexes[0].astype('int32')


@njit(
    'int32[:](DictType(unicode_type, float32[:, :]), unicode_type, int32)', nogil=True
)
def get_transaction_gte_field_indexes_jit(features, field, value):
    indexes = np.where(features[field] > value)
    return indexes[0].astype('int32')


class EqFeatureTransactionFilter(TransactionFilter):
    def as_jit_function(self):
        return get_transaction_eq_field_indexes_jit, (self.field, self.value)

    def get_features_need_for_filter(self) -> list[str]:
        return [self.field]

    def __init__(self, field: str, value: int):
        super().__init__()
        self.field = field
        self.value = np.int32(value)

    def get_transaction_indexes(self, features: dict[str, np.ndarray]) -> list[int]:
        return get_transaction_eq_field_indexes_jit(features, self.field, self.value)


class FeatureGTETransactionFilter(TransactionFilter):
    def as_jit_function(self):
        return get_transaction_gte_field_indexes_jit, (self.field, self.value)

    def get_features_need_for_filter(self) -> list[str]:
        return [self.field]

    def __init__(self, field: str, value: float):
        super().__init__()
        self.field = field
        self.value = np.int32(value)

    def get_transaction_indexes(self, features: dict[str, np.ndarray]) -> list[int]:
        return get_transaction_gte_field_indexes_jit(features, self.field, self.value)
