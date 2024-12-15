from __future__ import annotations
import datetime
from copy import copy
from datetime import datetime as dt
from typing import Dict

import numpy as np
import torch

from src.data.components.augmentation import VCompose
from src.data.components.boundary_generator import BoundaryGenerator
from src.data.components.target_generator import TargetGeneratorCompose
from src.data.components.target_mixer import TargetMixerCompose
from src.data.components.transforms import Compose


class TransactionDatasetDiskFile(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        indexes,
        targets: Dict,
        length_params: Dict,
        features: Dict,
        transforms: Compose,
        augmentations: VCompose = None,
        debug=False,
        dropout=0.0,
        shuffler=0.0,
        multi_task=False,
    ):
        self.data = data
        self.targets = targets
        self.indexes = indexes
        self.dropout = dropout
        if debug:
            self.indexes = data.get_indexes()
            self.targets = {app_id: 0.0 for app_id in self.indexes}
            for idx in self.indexes[: int(len(self.indexes) * 0.1)]:
                self.targets[idx] = 1.0

        self.max_days = length_params["max_days"]
        self.lag_days = length_params["lag_days"]
        self.max_seq_len = length_params["max_seq_len"]

        self.transforms = transforms
        self.features = {
            "categorical": [feature[0] for feature in features["categorical"]],
            "numerical": [feature[0] for feature in features["numerical"]],
        }
        self.numerical_features_len = sum([feature[1] for feature in features["numerical"]])

        self.augmentations = augmentations

        self.multi_task = multi_task

    def __len__(self):
        return len(self.indexes)

    @staticmethod
    def get_data(
        data, feature_names, window_start_pos, app_date_pos, max_seq_len, shift, feature_len
    ):
        length = app_date_pos - window_start_pos
        res = torch.zeros(
            min(max_seq_len, length) or 1,
            feature_len,
        )

        if length:
            res = []
            for idx, feature_name in enumerate(feature_names):
                feature = data[feature_name]

                arr = copy(feature[window_start_pos:app_date_pos])
                arr_tensor = torch.tensor(arr)
                tmp = arr_tensor[-max_seq_len:]
                feature_shape = tmp.shape
                if shift > 0:
                    res.append(
                        torch.cat([tmp[-shift:], tmp[:-shift]]).reshape(feature_shape[0], -1)
                    )
                else:
                    res.append(tmp.reshape(feature_shape[0], -1))
            res = torch.cat(res, dim=1)
        return res

    def __getitem__(self, idx):
        app_id = self.indexes[idx]

        if self.multi_task:
            target = [one_target.get(app_id, -1) for one_target in self.targets]
        else:
            target = self.targets[app_id]

        borrower = self.data[app_id]

        feature_arrays = borrower["feature_arrays"]

        if self.augmentations:
            feature_arrays = self.augmentations(feature_arrays)

        # for target transforms - add target to array, and delete after update
        feature_arrays["target"] = target
        feature_arrays = self.transforms(feature_arrays)
        target = feature_arrays.pop("target")

        if type(borrower["APPLICATION_DATE"]) == str:
            application_date = dt.strptime(
                borrower["APPLICATION_DATE"], "%Y-%m-%d"
            ) - datetime.timedelta(self.lag_days)
        else:
            application_date = borrower["APPLICATION_DATE"] - datetime.timedelta(self.lag_days)
        # days_before_application = [ for i in feature_arrays["local_date"]]
        # find left bound
        if self.max_days:
            first_trans_date = np.datetime64(application_date, "s") - np.timedelta64(
                self.max_days, "D"
            )
            window_start_pos = np.searchsorted(feature_arrays["local_date"], first_trans_date)
        else:
            window_start_pos = 0

        # find right bound
        app_date_pos = np.searchsorted(feature_arrays["local_date"], application_date)

        shift = 0
        cat_features = self.get_data(
            feature_arrays,
            self.features["categorical"],
            window_start_pos,
            app_date_pos,
            self.max_seq_len,
            shift,
            len(self.features["categorical"]),
        )

        if len(self.features["numerical"]) > 0:
            num_features = self.get_data(
                feature_arrays,
                self.features["numerical"],
                window_start_pos,
                app_date_pos,
                self.max_seq_len,
                shift,
                self.numerical_features_len,
            )
        else:
            num_features = torch.zeros((len(cat_features), 0)).float()

        return {
            "numerical": num_features.float(),
            "categorical": cat_features.long(),
            "targets": target,
            "sample_idx": app_id,
        }


class TransactionDatasetWithSyntheticTargetDiskFile(TransactionDatasetDiskFile):
    def __init__(
            self,
            boundary_generator: BoundaryGenerator,
            target_generator: TargetGeneratorCompose,
            target_mixer: TargetMixerCompose,
            *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.boundary_generator = boundary_generator
        self.target_generator = target_generator
        self.target_mixer = target_mixer

    def __getitem__(self, idx):
        app_id = self.indexes[idx]
        borrower = self.data[app_id]

        feature_arrays = borrower["feature_arrays"]

        if self.augmentations:
            feature_arrays = self.augmentations(feature_arrays)

        feature_arrays = self.transforms(feature_arrays)

        if type(borrower["APPLICATION_DATE"]) == str:
            application_date = dt.strptime(
                borrower["APPLICATION_DATE"], "%Y-%m-%d"
            ) - datetime.timedelta(self.lag_days)
        else:
            application_date = borrower["APPLICATION_DATE"] - datetime.timedelta(self.lag_days)

        # find left bound
        if self.max_days:
            first_trans_date = np.datetime64(application_date, "s") - np.timedelta64(
                self.max_days, "D"
            )
            window_start_pos = np.searchsorted(feature_arrays["local_date"], first_trans_date)
        else:
            window_start_pos = 0

        # find right bound
        app_date_pos = np.searchsorted(feature_arrays["local_date"], application_date)

        shift = 0

        time_splitting_indices = self.boundary_generator.get_interval_boundaries(feature_arrays)
        targets = self.target_generator.generate_targets(
            feature_arrays, time_splitting_indices
        )
        targets = self.target_mixer.mix_targets(targets)

        # todo: for debug
        time_feature = self.boundary_generator.get_time_feature(feature_arrays)
        feature_arrays = dict(feature_arrays)
        feature_arrays['days_before_application'] = np.log(time_feature + 1)

        cat_features = self.get_data(
            feature_arrays,
            self.features["categorical"],
            window_start_pos,
            app_date_pos,
            self.max_seq_len,
            shift,
            len(self.features["categorical"]),
        )

        if len(self.features["numerical"]) > 0:
            num_features = self.get_data(
                feature_arrays,
                self.features["numerical"],
                window_start_pos,
                app_date_pos,
                self.max_seq_len,
                shift,
                self.numerical_features_len,
            )
        else:
            num_features = torch.zeros((len(cat_features), 0)).float()

        return {
            "numerical": num_features.float(),
            "categorical": cat_features.long(),
            "targets": targets,
            "sample_idx": app_id,
            "target_mask": convert_indexes_into_mask(time_splitting_indices)
        }


class TransactionDatasetDiskFileSeq2Seq(TransactionDatasetDiskFile):
    def __init__(self, output_mask: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_mask = output_mask

    def __getitem__(self, idx):
        output: Dict = super().__getitem__(idx)
        app_id = self.indexes[idx]
        output_mask = self.output_mask[app_id]
        return {**output, "output_mask": output_mask}


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def convert_indexes_into_mask(indexes: np.ndarray[
    int
]):
    length = max(indexes) + 1
    mask = np.full(length, fill_value=False, dtype=bool)
    mask[indexes] = True
    return mask
