from abc import ABC

import numpy as np
from typing import Sequence, Dict


class Transform(ABC):
    def setup(self) -> None:
        pass

    def __call__(self, entity: Dict) -> Dict:
        raise NotImplementedError


class FieldTransform(Transform, ABC):
    def __init__(self, field_to_apply, field_to_write_transform_result):
        self.field_to_apply = field_to_apply
        self.field_to_write_transform_result = field_to_write_transform_result

    def transform(self, array):
        raise NotImplementedError

    def __call__(self, entity: Dict) -> Dict:        
        value = entity[self.field_to_apply]
        entity[self.field_to_write_transform_result] = self.transform(value)
        return entity


class Compose:
    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = transforms

    def setup(self):
        for transform in self.transforms:
            transform.setup()

    def __call__(self, entity: Dict) -> Dict:
        for transform in self.transforms:
            entity = transform(entity)
        return entity


class Log10ClipShiftTransform(FieldTransform):
    def __init__(self, shift=100., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift = shift

    def transform(self, array: np.array) -> np.array:
        signs = np.sign(array)
        signed_log_array = np.log10(np.abs(array) + self.shift) * signs
        return signed_log_array
