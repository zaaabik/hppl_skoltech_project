from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum

import torch

from src.dataset_generate.src.transaction import AppFeaturesBase


class TaskType(str, Enum):
    """Task type."""

    regression = "categorical"
    categorical = "numerical"


class TargetGenerator(ABC):
    """Class for generation target for transaction."""

    @abstractmethod
    def get_target(self, app_features: AppFeaturesBase) -> float | int:
        """Return target based on app_features."""
        raise NotImplementedError


class AmountReducer:
    def __init__(self, reduce_function: Callable):
        """Return target based on app_features."""
        self.reduce_function = reduce_function

    def __call__(self, amounts: torch.Tensor) -> float | int:
        """
        Params:
            amounts (torch.Tensor): amounts using for compute target
        Returns:
            (int | float) target for regression or categorical
        """
        return self.reduce_function(amounts)


class MCCAmountThrTarget(TargetGenerator):
    def __init__(self, target_mcc: int, amount_reducer: Callable, thr: float):
        """
        Args:
            target_mcc (int): mcc code of target transaction
            amount_reducer (Callable): function reduce amount of transaction with target mcc
            thr (float): threshold to make binary target from float number
        """
        super().__init__()
        self.target_mcc = target_mcc
        self.amount_reducer = amount_reducer
        self.thr = thr

    def get_target(self, app_features: AppFeaturesBase) -> int:
        """Reduce amount of transaction with target mcc and compare with threshold.

        Params:
            app_features (AppFeaturesBase): app transactions
        Returns:
            (int) binary target
        """
        needed_transaction = app_features.mcc_code == self.target_mcc
        needed_amount = app_features.amount_rur[needed_transaction]
        reduced_amount = self.amount_reducer(needed_amount)
        return int(reduced_amount > self.thr)


class MCCAmountRegressionTarget(TargetGenerator):
    def __init__(self, target_mcc: int, amount_reducer: Callable):
        """
        Args:
            target_mcc (int): mcc code of target transaction
            amount_reducer (Callable): function reduce amount of transaction with target mcc
        """
        super().__init__()
        self.target_mcc = target_mcc
        self.amount_reducer = amount_reducer

    def get_target(self, app_features: AppFeaturesBase) -> float:
        """Reduce amount of transaction with target mcc and compare with threshold.

        Params:
            app_features (AppFeaturesBase): app transactions
        Returns:
            (int) binary target
        """
        needed_transaction = app_features.mcc_code == self.target_mcc
        needed_amount = app_features.amount_rur[needed_transaction]
        reduced_amount = self.amount_reducer(needed_amount)
        return float(reduced_amount)


class MultiMCCAmountRegressionTargetSum(TargetGenerator):
    def __init__(self, target_mcc_1: int, target_mcc_2: int):
        """
        Args:
            target_mcc (int): mcc code of target transaction
            amount_reducer (Callable): function reduce amount of transaction with target mcc
        """
        super().__init__()
        self.target_mcc_1 = target_mcc_1
        self.target_mcc_2 = target_mcc_2

    def get_target(self, app_features: AppFeaturesBase) -> float:
        """Reduce amount of transaction with target mcc and compare with threshold.

        Params:
            app_features (AppFeaturesBase): app transactions
        Returns:
            (int) binary target
        """
        needed_transaction = app_features.mcc_code == self.target_mcc_1
        needed_amount_mcc_1 = app_features.amount_rur[needed_transaction]

        needed_transaction = app_features.mcc_code == self.target_mcc_2
        needed_amount_mcc_2 = app_features.amount_rur[needed_transaction]

        reduced_amount = torch.sum(needed_amount_mcc_1) + torch.sum(needed_amount_mcc_2)

        return float(reduced_amount)


class SlidingWindowTargetGenerator(ABC):
    """Class for generation target for transaction."""

    @abstractmethod
    def get_target(self, app_features: AppFeaturesBase) -> torch.Tensor:
        """Return target based on app_features for each transaction produce target."""
        raise NotImplementedError


class ConstantSlidingWindowTargetGenerator(SlidingWindowTargetGenerator):
    """Sliding window target for each element."""

    def __init__(
        self,
        target_mcc: int,
        window_size: int,
        reduce: str,
        fill_na: bool = True,
        fill_na_value: float = 0.0,
    ):
        """
        Args:
            target_mcc (int): mcc code of target transaction
            window_size (int): size of window to calculate target
            reduce (str): function reduce amount of transaction with target mcc
        """
        super().__init__()
        self.target_mcc = target_mcc
        self.reduce = reduce
        assert window_size % 2 == 1, "window size should be odd"

        self.window_size = window_size

        self.fill_na = fill_na
        self.fill_na_value = fill_na_value

    def pad_tensor_for_sliding_window_wo_look_into_future(
        self, tensor: torch.Tensor, value: float = 0
    ) -> torch.Tensor:
        """Add padding from left and right to be able to calculate convolution for start and end
        elements.

        Params:
            tensor (torch.Tensor): tensor to apply padding
        Returns:
            (torch.Tensor) tensor with padding
        """
        return torch.nn.functional.pad(tensor, (self.window_size - 1, 0), value=value)

    def fill_tensor_na(self, tensor: torch.Tensor) -> torch.Tensor:
        """Replace nan in tensors It could be very helpful in case when you aggregate in window
        where no target element.

        Params:
            tensor (torch.Tensor): tensor to apply filling nan
        Returns:
            (torch.Tensor) filled tensor
        """
        return torch.nan_to_num(tensor, nan=self.fill_na_value, neginf=self.fill_na_value)

    def get_target(self, app_features: AppFeaturesBase) -> torch.Tensor:
        """Get target for each element of the sequence. Main idea of implementation is to do
        convolution and then do shift of results, restrict looking into elements from future.

        Example:
            [1,2,3,4], window 3
            [0,0,1,2,3,4] - apply padding (def pad_tensor_for_sliding_window_wo_look_into_future)
            [1,3,6,9,7] - apply convolution with 1 in weights wo padding and stride 1
            [1,3,6,9] - remove last element (window size // 2 - 1)

        Params:
            app_features (AppFeaturesBase): app transactions
        Returns:
            (int) binary target
        """

        is_mcc_target_marker = (app_features.mcc_code == self.target_mcc).float()
        filtered_mcc = app_features.amount_rur * is_mcc_target_marker
        weight = torch.ones(self.window_size).float()
        padded_filtered_mcc = self.pad_tensor_for_sliding_window_wo_look_into_future(filtered_mcc)

        with torch.no_grad():
            if self.reduce == "sum":
                result = torch.conv1d(padded_filtered_mcc[None], weight[None, None])[0]
                # return result
            elif self.reduce == "mean":
                sum_of_windowed_amount = torch.conv1d(
                    padded_filtered_mcc[None], weight[None, None]
                )[0]
                is_mcc_target_marker = self.pad_tensor_for_sliding_window_wo_look_into_future(
                    is_mcc_target_marker
                )
                count_of_element_in_window = torch.conv1d(
                    is_mcc_target_marker[None], weight[None, None]
                )[0]
                result = sum_of_windowed_amount / count_of_element_in_window
            elif self.reduce == "max":
                min_value = float("-inf")
                non_target_mcc = app_features.mcc_code != self.target_mcc
                filled_amount = torch.masked_fill(
                    app_features.amount_rur, non_target_mcc, min_value
                )
                padded_amount = self.pad_tensor_for_sliding_window_wo_look_into_future(
                    filled_amount, value=min_value
                )
                unfolded_amount = padded_amount.unfold(0, self.window_size, 1)
                max_for_each_window = unfolded_amount.max(dim=1).values
                result = max_for_each_window
            else:
                raise ValueError("reduce should be {sum, max, mean}")

            if self.fill_na:
                result = self.fill_tensor_na(result)

            return result


class GroupByTimeSlidingWindowTargetGenerator(SlidingWindowTargetGenerator):
    def __init__(
        self, target_mcc: int, reduce: Callable, fill_na_value: float = 0.0, days_lag: int = 0.0
    ):
        """
        Args:
            target_mcc (int): mcc code of target transaction
            reduce (Callable): function reduce amount of transaction with target mcc
            fill_na_value (float): value to fill window with no target transaction
            days_lag (int): lag between target and days
        """
        super().__init__()
        self.target_mcc = target_mcc
        self.reduce = reduce
        self.fill_na_value = fill_na_value
        self.days_lag = days_lag

    def filtered_idxes(self, prev_mcc, prev_time_feature, current_time_feature):
        return (prev_mcc == self.target_mcc) & (
                prev_time_feature == current_time_feature
        )

    def get_target(self, app_features: AppFeaturesBase) -> torch.Tensor:
        transaction_count = len(app_features.amount_rur)
        time_feature = app_features.days_before_application
        mcc = app_features.mcc_code
        amount = app_features.amount_rur
        targets = []
        for idx in range(transaction_count):
            prev_amount = amount[: idx + 1]
            prev_mcc = mcc[: idx + 1]
            prev_time_feature = time_feature[: idx + 1]

            current_time_feature = time_feature[idx] + self.days_lag
            filtered_idxes = self.filtered_idxes(prev_mcc, prev_time_feature, current_time_feature)
            if len(filtered_idxes) == 0:
                current_target = self.fill_na_value
            else:
                current_target = self.reduce(prev_amount[filtered_idxes])
            targets.append(current_target)
        return torch.tensor(targets)
