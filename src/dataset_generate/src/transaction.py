from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from random import shuffle
from typing import Any, List

import numpy
import numpy as np
import torch
from torch.distributions import Distribution

from src.dataset_generate.src.utils import merge_dataclasses_with_torch_tensors


@dataclass
class Transaction:
    """Class for hold single transaction features."""

    amount_rur: float
    mcc_code: int
    # days_before_application: int = 0

    # method to convert class int dict
    as_dict = asdict


class AmountDistribution(ABC):
    """Base class for sampling amount training.

    If you want to create custom amount distribution just create own class with sample method
    """

    @abstractmethod
    def sample(self) -> float:
        """Create one observation.

        Returns:
             (float) sample from random distribution
        """
        raise NotImplementedError


class ConditionalAmountDistribution(ABC):
    """Base class for sampling amount training.

    If you want to create custom amount distribution just create own class with sample method
    """

    @abstractmethod
    def conditional_sample(self, context: dict[Transaction] | None = None) -> float:
        """Create one observation.

        Args:
            context (List[Transaction]) : list of previous transactions for conditional generation
        Returns:
             (float) sample from random distribution
        """
        raise NotImplementedError


@dataclass
class AppTransactions:
    """Create one observation."""

    transactions: dict[Transaction]


@dataclass
class AppFeaturesBase:
    """Placeholder for all app features."""

    mcc_code: numpy.ndarray
    amount_rur: numpy.ndarray
    days_before_application: numpy.ndarray = None

    @classmethod
    def get_feature_list(cls):
        """Return list of app features."""
        field_names = [f.name for f in fields(cls)]
        sorted(field_names)
        return field_names

    @classmethod
    def from_feature_array(cls, feature_array: dict) -> AppFeaturesBase:
        days_before_application = None
        if "days_before_application" in feature_array:
            days_before_application = torch.tensor(feature_array["days_before_application"])
        return cls(
            mcc_code=np.array(feature_array["mcc_code"]),
            amount_rur=np.array(feature_array["amount_rur"]),
            days_before_application=np.array(days_before_application),
        )


class TransactionSampler:
    def __init__(self, amount_distribution: AmountDistribution | Distribution, mcc: int, **kwargs):
        """Create transaction with mcc code and random sample amount
        Params:
            amount_distribution (AmountDistribution): function to create random amount
            mcc (int): fixed mcc code
        """
        self.amount_distribution = amount_distribution
        self.mcc_code = mcc

        if kwargs.get("sign") is not None:
            self.sign = kwargs.get("sign")
        else:
            self.sign = 1

    def sample(self) -> Transaction:
        """Create transaction with mcc code and random sample amount.

        Returns:
             (Transaction) with defined mcc and sampled random amount
        """
        sampled_amount = float(self.amount_distribution.sample()) * self.sign
        return Transaction(amount_rur=sampled_amount, mcc_code=self.mcc_code)


class BaseAppTransactionSampler(ABC):
    @abstractmethod
    def sample(self) -> AppFeaturesBase:
        """Sample random app features."""
        raise NotImplementedError


class AppWeightedTransactionSampler(BaseAppTransactionSampler):
    """This class take samplers with probability and random sample **length** transactions, in each
    step we weighed choose sampler (MCC code) and the sample random fields and obtain
    transaction."""

    def __init__(
        self, transaction_sampler_probability: dict[tuple[TransactionSampler, float]], length: int
    ):
        """Imitate app creating transaction based on multiple samplers with probabilities
        Params:
            transaction_chain_mapper (List[Tuple[TransactionSampler, float]]): dict where key is class for creating
                transaction and values is probability of this transaction in chain of transactions
            length (int): number of sampled transaction
        Returns:
             (Transaction)
        """
        self.length = length
        self.samplers: list[Any] = []
        self.probabilities = []
        for sampler, probability in transaction_sampler_probability:
            self.samplers.append(sampler)
            self.probabilities.append(probability)
        assert np.allclose(sum(self.probabilities), 1.0)

    def sample(self) -> AppFeaturesBase:
        """
        Sample length random transaction using weighted samplers
        Returns:
             (Dict[str, Dict[str, torch.tensor]]) transactions
        """
        # Each transaction created by sampler, here we choose randomly sampler for each position using probabilities
        # as weights
        transaction_samplers = np.random.choice(
            self.samplers, size=self.length, p=self.probabilities
        )

        transactions = []
        reversed_indexes = list(range(len(transaction_samplers)))[::-1]
        for index, transaction_sampler in zip(reversed_indexes, transaction_samplers):
            # get transaction from sampler
            transaction = transaction_sampler.sample()
            # transaction.days_before_application = index
            transactions.append(transaction)

        output = AppFeaturesBase(**merge_dataclasses_with_torch_tensors(transactions))
        return output


class TransactionCountDistribution(ABC):
    @property
    @abstractmethod
    def max_count(self) -> int | None:
        """

        Returns:
             maximum possible number of generated transactions or None for fill the rest transaction
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> int | None:
        """
        Return number of transaction for app
        Returns:
             (int | None) number of transaction with certain mcc code.
             In case of None, transaction sampler should fill until the max_length
        """
        raise NotImplementedError


class UniformTransactionCountDistribution(TransactionCountDistribution):
    def __init__(self, lower: int, upper: int):
        super().__init__()
        self.lower = lower
        self.upper = upper

    @property
    def max_count(self) -> int:
        """
        Returns:
             (int) maximum possible number of generated transactions
        """
        return self.upper

    def sample(self) -> int:
        """
        Return number of transaction for app
        Returns:
             (int) number of transaction with certain mcc code
        """
        return torch.randint(low=self.lower, high=self.upper + 1, size=(1,)).item()


class RestTransactionFiller(TransactionCountDistribution):
    """Fill the rest of transaction with certain mcc code."""

    @property
    def max_count(self) -> None:
        """As convention this method should return None for filling rest of transactions."""
        return None

    def sample(self) -> None:
        """As convention this method should return None for filling rest of transactions."""
        return None


class AppCounterTransactionSampler(BaseAppTransactionSampler):
    """This class sample app transaction based on sampler and number of transaction produced by
    this sampler."""

    def __init__(
        self,
        transaction_sampler_counter: list[tuple[TransactionSampler, TransactionCountDistribution]],
        max_length: int,
        time_sampler: TimeFeatureGenerator | None = None,
    ):
        """Imitate app creating transaction based on multiple samplers with probabilities
        Params:
            transaction_sampler_counter (List[Tuple[TransactionSampler, int]]): list of tuples, each tuple contains
                sampler and number transaction produced by this sampler
            length (int): total number of sampled transaction
        """
        self.max_length = max_length
        self.samplers: list[Any] = []
        self.count_ranges = []
        max_possible_transactions_generated = 0
        self.time_sampler = time_sampler or NumberOfElementInSeqFeatureGenerator()

        number_of_rest_fillers = 0

        self.rest_filler_sampler = None

        for sampler, count_range in transaction_sampler_counter:
            if isinstance(count_range, RestTransactionFiller):
                number_of_rest_fillers += 1
                self.rest_filler_sampler = sampler
            else:
                self.samplers.append(sampler)
                self.count_ranges.append(count_range)
                max_possible_transactions_generated += count_range.max_count

        assert number_of_rest_fillers <= 1, "number of rest fillers should not greater than 1"
        assert (
            max_possible_transactions_generated <= max_length
        ), f"total number of transaction {max_possible_transactions_generated} greater then {max_length} "

    @staticmethod
    def _sample_transactions(
        transaction_sampler: TransactionSampler, length: int
    ) -> list[Transaction]:
        """
        Sample random transactions based on samplers
        Args:
             transaction_sampler (TransactionSampler)
             length (int)
        Return:F
            (List[Transaction])
        """
        transactions = []
        for _ in range(length):
            transaction = transaction_sampler.sample()
            transactions.append(transaction)
        return transactions

    def sample(self) -> AppFeaturesBase:
        """
        Sample random transactions based on samplers
        Returns:
             (AppFeaturesBase) transactions
        """

        transactions = []
        current_transaction_count = 0
        for count_ranges, transaction_sampler in zip(self.count_ranges, self.samplers):
            current_type_transaction_count = count_ranges.sample()
            current_transaction_count += current_type_transaction_count

            transactions.extend(
                self._sample_transactions(transaction_sampler, current_type_transaction_count)
            )

        if self.rest_filler_sampler:
            need_to_fill = self.max_length - current_transaction_count
            transactions.extend(self._sample_transactions(self.rest_filler_sampler, need_to_fill))
            current_transaction_count += need_to_fill

        assert current_transaction_count <= self.max_length

        shuffle(transactions)

        output = AppFeaturesBase(**merge_dataclasses_with_torch_tensors(transactions))

        output.days_before_application = torch.tensor(
            self.time_sampler.generate_time_feature(output)
        )

        return output


class TimeFeatureGenerator(ABC):
    """Class for generation time feature for transaction sequence."""

    @abstractmethod
    def generate_time_feature(self, app_features: AppFeaturesBase) -> list:
        """
        Args:
            app_features (AppFeaturesBase): application with transaction
        Returns:
            (torch.Tensor) value of time for each transaction
        """


class NumberOfElementInSeqFeatureGenerator(TimeFeatureGenerator):
    """Generate time features as position in a sequence with reverse order [6,5,4,3,2,1,0]"""

    def generate_time_feature(self, app_features: AppFeaturesBase) -> list:
        """
        Args:
            app_features (AppFeaturesBase): application features
        Return:
            (torch.Tensor) time feature as reversed range from zero to number of transaction
        """
        transaction_count = len(app_features.amount_rur)
        return list(range(0, transaction_count))[::-1]


class MultipleTransactionPerDayTimeFeatureGenerator(TimeFeatureGenerator):
    """Class for generation time feature for transaction sequence.

    Generate sequence of day idx for each transaction days_before_application = [1,1,1,2,2,3,3]

    min_transaction_in_a_day, max_transaction_in_a_day are hyperparameters for each day we sample
    number of transaction randint(min_transaction_in_a_day, max_transaction_in_a_day)
    """

    def __init__(self, min_transaction_in_a_day: int, max_transaction_in_a_day: int):
        """
        Args:
            min_transaction_in_a_day (int): min possible transaction in a day
            max_transaction_in_a_day (int): max possible transaction in a day
        """
        self.min_transaction_in_a_day = min_transaction_in_a_day
        self.max_transaction_in_a_day = max_transaction_in_a_day

    def generate_time_feature(self, app_features: AppFeaturesBase) -> list:
        transaction_count = len(app_features.amount_rur)

        current_transaction_idx = 0
        current_day = 0

        days_before_application = []
        while current_transaction_idx < transaction_count:
            transaction_in_this_day = np.random.randint(
                self.min_transaction_in_a_day, self.max_transaction_in_a_day + 1
            )
            # days without transaction should be skipped
            if transaction_in_this_day > 0:
                days_before_application.extend([current_day] * transaction_in_this_day)
            current_day += 1
            current_transaction_idx += transaction_in_this_day

        # days_before_application = torch.concat(days_before_application)

        # reverse days to be 0 day close to application day
        days_before_application = days_before_application[::-1]

        # in a last step we can generate more time feature than we need
        # and here we remove features to be the same length as amount
        days_before_application = days_before_application[:transaction_count]
        return list(days_before_application)
