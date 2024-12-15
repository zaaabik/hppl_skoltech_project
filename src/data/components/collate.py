from __future__ import annotations

from abc import ABC
from collections import namedtuple
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

ModelBatch_ = namedtuple(
    "ModelBatch", ["numerical", "categorical", "targets", "mask", "sample_idxes", "output_mask"]
)


class ModelBatch(ModelBatch_):
    def __new__(
        cls,
        numerical: Optional[torch.FloatTensor] = None,
        categorical: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        sample_idxes: List[str] = None,
        output_mask: Optional[torch.Tensor] = None,
    ):
        return super().__new__(
            cls,
            numerical=numerical,
            categorical=categorical,
            targets=targets,
            mask=mask,
            sample_idxes=sample_idxes,
            output_mask=output_mask,
        )


@dataclass
class ModelMultiTargetBatch:
    numerical: Optional[torch.FloatTensor] = None,
    categorical: Optional[torch.Tensor] = None,
    targets: Optional[dict[str, torch.Tensor]] = None,
    mask: Optional[torch.Tensor] = None,
    sample_idxes: List[str] = None,
    target_mask: Optional[torch.Tensor] = None,


def create_padding_mask(length: Sequence[int], max_length: int) -> torch.BoolTensor:
    """Creates padding mask for sequence.

    Creates len(length) x max_length BoolTensor, where each row represents sentence.
    Tensor has False values at indexes <= length[i] for i'th sequence.

    Args:
        length: lengths of sequences, list[int].
        max_length: maximal length of sequences, int.

    Returns:
        mask, BoolTensor.
    """
    device = 'cpu'
    if isinstance(length, torch.Tensor):
        device = length.device
    mask = torch.arange(max_length, device=device)[None, :] >= torch.tensor(length, device=device)[:, None]
    return mask


class Collator(ABC):
    def __call__(self, batch: List[Dict]) -> Dict:
        raise NotImplementedError


class BaseCollator(Collator, ABC):
    """Collator for creating batch (ModelBatch).

    This class takes list of dicts where dict is returned from __getitem__ function of Dataset class.
    It pads all input tensors to seq_len.

    Attributes:
        seq_len: same as max_seq_len, int.
    """

    seq_len = None

    def __init__(self, seq_len: int):
        self.seq_len = seq_len

    def pad_tensors(self, data: List[Dict], field: str, additional_field: str = None):
        if additional_field is None:
            fields_to_pad = [
                torch.tensor(item[field]) for item in data
            ]  # list of tensors obtained by taking every field key from list of dicts
        else:
            fields_to_pad = [
                torch.tensor(item[field][additional_field]) for item in data
            ]  # list of tensors obtained by taking every field key from list of dicts
        length = [len(item[field]) for item in data]
        padded_data = torch.nn.utils.rnn.pad_sequence(
            fields_to_pad, batch_first=True
        )  # pads all of torch tensors to seq_len]
        padded_data = torch.nn.functional.pad(
            padded_data, (0, 0, 0, self.seq_len - padded_data.shape[1])
        )  # adds seq_len - x.shape[1] tokens at the end to achieve seq_len
        return padded_data, length

    def __call__(self, data: List[Dict]) -> ModelBatch:
        numerical, _ = self.pad_tensors(data, "numerical")
        categorical, length = self.pad_tensors(data, "categorical")

        targets = torch.tensor([item["targets"] for item in data], dtype=torch.float32)[:, None]
        sample_idxes = [item["sample_idx"] for item in data]
        mask = create_padding_mask(length, self.seq_len)

        return ModelBatch(
            numerical=numerical.float(),
            categorical=categorical.long(),
            targets=targets,
            sample_idxes=sample_idxes,
            mask=mask,
        )


class BaseCollatorDropout(Collator, ABC):
    """Collator for creating batch (ModelBatch).

    This class takes list of dicts where dict is returned from __getitem__ function of Dataset class.
    It pads all input tensors to minimum of seq_len and maximum length of input sequences.

    Attributes:
        seq_len: same as max_seq_len, int.
    """

    seq_len = None

    def __init__(self, seq_len: int):
        self.seq_len = seq_len

    def pad_tensors(self, data: List[Dict], field: str):
        fields_to_pad = [
            item[field] for item in data
        ]  # list of tensors obtained by taking every field key from list of dicts
        length = [len(item[field]) for item in data]
        padded_data = torch.nn.utils.rnn.pad_sequence(
            fields_to_pad, batch_first=True
        )  # pads list of torch tensors to max tensor len (max L same as max x.shape[1])
        if padded_data.shape[1] > self.seq_len:
            padded_data = torch.nn.functional.pad(
                padded_data, (0, 0, 0, self.seq_len - padded_data.shape[1])
            )  # cuts from the beginning seq_len - x.shape[1] tokens to achieve seq_len
        return padded_data, length

    def __call__(self, data: List[Dict]) -> ModelBatch:
        numerical, _ = self.pad_tensors(data, "numerical")
        categorical, length = self.pad_tensors(data, "categorical")

        targets = torch.tensor([item["targets"] for item in data], dtype=torch.float32)[:, None]
        sample_idxes = [item["sample_idx"] for item in data]
        mask = create_padding_mask(length, numerical.shape[1])

        return ModelBatch(
            numerical=numerical.float(),
            categorical=categorical.long(),
            targets=targets,
            sample_idxes=sample_idxes,
            mask=mask,
        )


class SequenceTargetCollator(BaseCollator):
    """Collator for creating batch (ModelBatch) for sequential target.

    This class takes list of dicts where dict is returned from __getitem__ function of Dataset class.
    It pads all input tensors to seq_len.

    Attributes:
        seq_len: same as max_seq_len, int.
    """

    def __call__(self, data: List[Dict]) -> ModelBatch:
        numerical, _ = self.pad_tensors(data, "numerical")
        output_mask, _ = self.pad_tensors(data, "output_mask")
        targets, _ = self.pad_tensors(data, "targets")
        categorical, length = self.pad_tensors(data, "categorical")

        sample_idxes = [item["sample_idx"] for item in data]
        mask = create_padding_mask(length, self.seq_len)

        return ModelBatch(
            numerical=numerical.float(),
            categorical=categorical.long(),
            targets=targets,
            sample_idxes=sample_idxes,
            mask=mask,
            output_mask=output_mask,
        )


class SequenceTargetCollatorMultiTarget(BaseCollator):
    """Collator for creating batch (ModelBatch) for sequential target.

    This class takes list of dicts where dict is returned from __getitem__ function of Dataset class.
    It pads all input tensors to seq_len.

    Attributes:
        seq_len: same as max_seq_len, int.
    """

    def pad_targets(self, data: List[Dict], field: str, additional_field: str = None):
        fields_to_pad = [
            torch.tensor(item[field][additional_field])[:, None] for item in data
        ]
        length = [len(item[field][additional_field]) for item in data]

        padded_data = torch.nn.utils.rnn.pad_sequence(
            fields_to_pad, batch_first=True
        )
        return padded_data, length

    def __call__(self, data: List[Dict]) -> ModelMultiTargetBatch:
        numerical, _ = self.pad_tensors(data, "numerical")
        target_names = list(data[0]["targets"].keys())

        tensor_target_dict: dict[torch.Tensor] = {}
        for target_name in target_names:
            tensor_target_dict[target_name], target_length = self.pad_targets(data, "targets", target_name)

        categorical, length = self.pad_tensors(data, "categorical")
        target_mask, _ = self.pad_tensors(data, "target_mask")

        sample_idxes = [item["sample_idx"] for item in data]
        mask = create_padding_mask(length, self.seq_len)
        # output_mask = create_padding_mask(target_length, self.seq_len)

        return ModelMultiTargetBatch(
            numerical=numerical.float(),
            categorical=categorical.long(),
            targets=tensor_target_dict,
            sample_idxes=sample_idxes,
            mask=mask,
            target_mask=target_mask,
        )
