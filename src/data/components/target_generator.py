from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numba
from numba import types as nbt
from numba import njit

from src.data.components.target_aggregation import FeatureAggregation
from src.data.components.transaction_filter import TransactionFilter


class BaseTargetModifier(ABC):
    @abstractmethod
    def generate_target(self,
                        features: dict[str, np.ndarray],
                        time_splitting_indices: np.ndarray[int]
                        ) -> (
            str, np.ndarray
    ):
        pass


class TargetGenerator(BaseTargetModifier, ABC):
    pass


class TargetGeneratorCompose:
    def __init__(self, target_generators: list[BaseTargetModifier]):
        self.target_generators = target_generators

    def __len__(self):
        return len(self.target_generators)

    def generate_targets(self, features: dict[str, np.ndarray],
                         time_splitting_indices: np.ndarray[int]) -> dict[str, np.ndarray]:
        features = {
            str(k): v.astype('float32') for k, v in features.items() if k != 'local_date'
        }
        features = get_numba_dict_from_dict(features)
        targets = {}
        for idx in range(len(self.target_generators)):
            target_generator = self.target_generators[idx]
            target_name, target = target_generator.generate_target(
                features, time_splitting_indices
            )
            targets[target_name] = target
        return targets


class AggregateTargetGenerator(TargetGenerator):
    def __init__(self,
                 transaction_filter: TransactionFilter,
                 aggregation_function: FeatureAggregation,
                 target_name: str,
                 look_back: int = 1
                 ):
        super().__init__()
        self.transaction_filter = transaction_filter
        self.aggregation_function = aggregation_function
        self.look_back = np.int32(look_back)
        self.target_name = target_name

    def generate_target(self, features: dict[str, np.ndarray],
                        time_splitting_indices: np.ndarray[int]) -> (
            dict[str, np.ndarray],
            np.ndarray[int],
            dict[str, np.ndarray]
    ):
        _features_need_to_aggregate = self.aggregation_function.get_features_need_for_aggregate() + \
                                      self.transaction_filter.get_features_need_for_filter()

        features_need_to_aggregate = numba.typed.List()
        [features_need_to_aggregate.append(x) for x in _features_need_to_aggregate]

        filter_function, filter_function_params = \
            self.transaction_filter.as_jit_function()

        aggregation_function, aggregation_function_params = self.aggregation_function.as_jit_function()

        aggregate_result = generate_target_from_intervals(
            features=features,
            time_splitting_indices=time_splitting_indices,
            features_need_to_aggregate=features_need_to_aggregate,
            aggregation_function=aggregation_function,
            aggregation_function_params=aggregation_function_params,
            filter_function=filter_function,
            filter_function_params=filter_function_params,
            look_back=self.look_back
        )

        return self.target_name, aggregate_result


def get_numba_dict_from_dict(my_dict):
    output = numba.typed.Dict.empty(
        key_type=numba.types.unicode_type,
        value_type=numba.types.float32[:, :]
    )

    for k, v in my_dict.items():
        output[k] = v
    return output


@njit(nogil=True)
def get_slices(features: numba.typed.Dict,
               features_need_to_aggregate,
               start: int, end: int):
    output_dict = numba.typed.Dict.empty(
        key_type=nbt.unicode_type,
        value_type=nbt.float32[:, :]
    )

    for feature_name in features_need_to_aggregate:
        window = features[feature_name][start:end]
        output_dict[feature_name] = window
    return output_dict


@njit(nogil=True, fastmath=True)
def generate_target_from_intervals(features,
                                   time_splitting_indices,
                                   features_need_to_aggregate,
                                   filter_function, filter_function_params, aggregation_function,
                                   aggregation_function_params, look_back=1):
    num_intervals = len(time_splitting_indices)
    current_target = np.zeros(num_intervals, dtype=np.float32)
    for index in range(num_intervals):
        start_index = max((index - look_back), 0)
        end_index = index
        start_transaction_index = time_splitting_indices[start_index] + 1
        if index == 0:
            start_transaction_index = 0
        end_transaction_index = time_splitting_indices[end_index]
        end_transaction_index += 1

        current_window_features = get_slices(features=features,
                                             features_need_to_aggregate=features_need_to_aggregate,
                                             start=start_transaction_index,
                                             end=end_transaction_index)
        target_transaction_indexes = filter_function(
            current_window_features, *filter_function_params
        )

        for feature_name in features_need_to_aggregate:
            current_window_features[feature_name] = current_window_features[feature_name][target_transaction_indexes]

        target = aggregation_function(
            current_window_features,
            *aggregation_function_params
        )
        current_target[index] = target
    return current_target