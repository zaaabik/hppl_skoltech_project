from abc import ABC

from typing import Sequence, Dict, List, Iterable, Optional, Any


class Augmentation(ABC):
    def __call__(self, features: Dict) -> Dict:
        raise NotImplementedError

    @staticmethod
    def get_sequence_length(features: Dict) -> int:
        return next(features.values().__iter__()).size


class VCompose(Augmentation):
    augmentations: Iterable[Augmentation]

    def __init__(self, augmentations: Iterable[Augmentation]):
        self.augmentations = augmentations

    def __call__(self, features: Dict) -> Dict:
        for augmentation in self.augmentations:
            features = augmentation(features)
        return features

    
class DoNothingAugmenter(Augmentation):
    def __call__(self, features: Dict) -> Dict:
        return features
