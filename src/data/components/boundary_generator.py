from __future__ import annotations

import datetime
import math
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from numba import njit

MAX_DAYS_IN_WEEK = 7
MAX_DAYS_IN_MONTH = 32
MAX_DAYS_IN_QUARTER = MAX_DAYS_IN_MONTH * 3
MAX_DAYS_IN_YEAR = MAX_DAYS_IN_MONTH * 12
MAX_WEEKS_IN_YEAR = 54


class BoundaryGenerator(ABC):
    """Base class to extract boundary of intervals"""

    @abstractmethod
    def get_interval_boundaries(self, features: dict[str, np.ndarray]) -> np.ndarray[int]:
        """
        Return boundary based on features
        Args:
            features (dict[str, np.ndarray]): dict of features each field is [L,d], L - length of seq,
                d - size of field (often 1, but for PLE could be greater)
        Returns:
            list[int], each element of list is position of interval end (day, month etc)
        """


def get_boundary_by_change_value_in_seq(feature: np.ndarray) -> np.ndarray:
    time_diff = feature[1:] - feature[:-1]
    change_day = time_diff != 0
    change_day = np.hstack((change_day, np.array([True])))
    return change_day


def day_extractor(time_feature: datetime.datetime) -> int:
    item = time_feature
    return item.day + item.month * MAX_DAYS_IN_MONTH + item.year * MAX_DAYS_IN_YEAR


def month_extractor(time_feature: datetime.datetime) -> int:
    item = time_feature
    return item.month * MAX_DAYS_IN_MONTH + item.year * MAX_DAYS_IN_YEAR


def quarter_extractor(time_feature: datetime.datetime) -> int:
    item = time_feature
    quarter = math.ceil(item.month / 3.)
    return quarter * MAX_DAYS_IN_QUARTER + item.year * MAX_DAYS_IN_YEAR


def year_extractor(time_feature: datetime.datetime) -> int:
    item = time_feature
    return item.year


def week_extractor(time_feature: datetime.datetime) -> int:
    item = time_feature
    week_of_the_year = item.isocalendar()[1]
    return item.year * MAX_WEEKS_IN_YEAR + week_of_the_year * MAX_DAYS_IN_WEEK


def time_feature_extractor_func(feature_name: str) -> Callable:
    if feature_name == 'day':
        return day_extractor
    if feature_name == 'week':
        return week_extractor
    elif feature_name == 'month':
        return month_extractor
    elif feature_name == 'quarter':
        return quarter_extractor
    elif feature_name == 'year':
        return year_extractor
    else:
        raise ValueError(f'Time extractor does not support {feature_name} extraction')


def convert_mask_to_indexes(
        last_transaction_of_time_interval_mask: np.ndarray[bool]
) -> np.ndarray[int]:
    return np.where(last_transaction_of_time_interval_mask)[0]


class TimeBoundaryGenerator(BoundaryGenerator):
    def __init__(self, time_feature: str,
                 local_date_feature_name: str = 'local_date'):
        """
        Generate time field for each transaction
        Args:
            time_feature (str): name of time field
            local_date_feature_name (str): field name of local date in each transaction
        Returns:
            (np.ndarray): time field of each element in seq
        """
        super().__init__()
        self.time_feature = time_feature
        self.local_date_feature_name = local_date_feature_name

    def _generate_time_feature(self,
                               time_features: np.ndarray[np.datetime64]
                               ) -> np.ndarray:
        """
        Generate time field for each transaction
        Args:
            time_features (np.ndarray[np.datetime64]): time field
        Returns:
            (np.ndarray): time field of each element in seq
        """
        extractor_func = time_feature_extractor_func(self.time_feature)

        extractor_func_vectorize = np.vectorize(extractor_func)
        return extractor_func_vectorize(time_features.astype(object))

    def get_time_feature(self, features):
        time_feature = self._generate_time_feature(
            features[self.local_date_feature_name]
        )
        return time_feature

    def get_interval_boundaries(self, features: dict[str, np.ndarray]) -> np.ndarray[int]:
        time_feature = self._generate_time_feature(
            features[self.local_date_feature_name]
        )
        boundary_mask = get_boundary_by_change_value_in_seq(time_feature)
        boundary_indexes = convert_mask_to_indexes(boundary_mask)
        return boundary_indexes.astype('int32')
