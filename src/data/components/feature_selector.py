from typing import Iterator, List, Union, Dict, Tuple, Sequence, Optional


class FeatureSelector:
    def __init__(self, categorical, numerical):
        # can't be passed simultaneously
        assert not (categorical['select_mask'] and categorical['select_indxs'] and categorical['select_names']), "can't be passed simultaneously"
        assert not (numerical['select_mask'] and numerical['select_indxs'] and numerical['select_names']), "can't be passed simultaneously"

        self.categorical = categorical
        self.numerical = numerical

    @staticmethod
    def create_mask(field):
        features_mask = [False] * len(field['input_features'])

        if field['select_mask']:
            features_mask = field['select_mask']
            assert len(field['select_mask']) == len(field['input_features']), "mask size should be equal num features"
        elif field['select_indxs']:
            for i in field['select_indxs']:
                features_mask[i] = True
        elif field['select_names']:
            for idx, name in enumerate(field['input_features']):
                if name[0] in field['select_names']:
                    features_mask[idx] = True
        else:
            features_mask = [True] * len(field['input_features'])

        field['mask'] = features_mask

    @staticmethod
    def select_output_features(field):
        field['output_features'] = [feature for feature, mask in zip(field['input_features'], field['mask']) if mask]

    def setup(self, features: Dict):
        self.numerical['input_features'] = features['numerical']
        self.categorical['input_features'] = features['categorical']

        self.create_mask(self.categorical)
        self.create_mask(self.numerical)

        self.select_output_features(self.categorical)
        self.select_output_features(self.numerical)

    def get_output_features(self):
        return {
            'numerical': self.numerical['output_features'],
            'categorical': self.categorical['output_features']
        }
