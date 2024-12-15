from abc import ABC
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

import pickle


class BaseTargetFileReader(ABC):
    def read(self, path) -> Tuple[Any, str]:
        raise NotImplementedError


class IndexReader:
    train_hash: str = ''
    val_hash: str = ''
    test_hash: str = ''
    predict_hash: str = ''

    def __init__(
        self,
        train_path: Optional[str or Tuple[str]] = None,
        test_path: str = None,
        val_path: Optional[str or Tuple[str]] = None,
        predict_path: str = None,
        reader: BaseTargetFileReader = None
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.predict_path = predict_path
        self.reader = reader

    @property
    def train_indexes(self):
        if isinstance(self.train_path, str):
            train_indexes, self.train_hash = self.reader.read(self.train_path)
        else:
            train_indexes, self.train_hash = zip(*(self.reader.read(train_path) for train_path in self.train_path))
        return train_indexes

    @property
    def test_indexes(self):
        test_targets, self.test_hash = self.reader.read(self.test_path)
        return test_targets

    @property
    def val_indexes(self):
        if isinstance(self.val_path, str):
            val_indexes, self.val_hash = self.reader.read(self.val_path)
        else:
            val_indexes, self.val_hash = zip(*(self.reader.read(val_path) for val_path in self.val_path))
        return val_indexes

    @property
    def predict_indexes(self):
        predict_indexes, self.predict_hash = self.reader.read(self.predict_path)
        return predict_indexes


class TargetReader:
    _targets: Dict = None
    _hash_sum: str = ''

    def __init__(
            self,
            targets_path: str = None,
            reader: BaseTargetFileReader = None
    ):
        super().__init__()
        self.reader = reader
        self.targets_path = targets_path

    def setup(self):
        self._targets, self._hash_sum = self.reader.read(self.targets_path)

    @property
    def targets(self):
        return self._targets

    @property
    def hash_sum(self):
        return self._hash_sum


class TargetReaderMultiTask:
    _targets: List[Dict] = []
    _hash_sum: str = ""

    def __init__(
            self,
            targets_path: List[str] = None,
            reader: BaseTargetFileReader = None
    ):
        super().__init__()
        self.reader = reader
        self.targets_path = targets_path

    def setup(self):
        for path in self.targets_path:
            targets, hash_sum = self.reader.read(path)
            self._targets.append(targets)
            self._hash_sum += hash_sum

    @property
    def targets(self):
        return self._targets

    @property
    def hash_sum(self):
        return self._hash_sum


def read_target_pickle_to_df(path) -> Dict:
    df = pd.read_pickle(path)
    return df.to_dict('index')
