import cProfile
import rootutils
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.augmentation import VCompose
from src.data.components.collate import SequenceTargetCollatorMultiTarget
from src.data.components.dataset import TransactionDatasetWithSyntheticTargetDiskFile
from src.data.components.transforms import Log10ClipShiftTransform, Compose
from src.data.components.boundary_generator import TimeBoundaryGenerator
from src.data.components.target_aggregation import FuncFeatureAggregation, sum_jit, mean_jit, max_jit, min_jit
from src.data.components.target_generator import TargetGeneratorCompose, AggregateTargetGenerator
from src.data.components.target_mixer import LinearBaseTargetMixer, TargetMixerCompose, DividerBaseTargetMixer
from src.data.components.transaction_filter import EqFeatureTransactionFilter, FeatureGTETransactionFilter
from tests.utils import generate_synthetic_data


length = 4096
num_application = 2000


def test_pytorch_dataloader_multiple_targets():
    num_targets = 20

    data = generate_synthetic_data(
        length=length, min_diff_day=1, max_diff_day=1, number_mcc=1
    )
    aggregate_target_generators = [
        AggregateTargetGenerator(
            transaction_filter=EqFeatureTransactionFilter(
                field='mcc_code', value=0
            ),
            aggregation_function=FuncFeatureAggregation(
                field='amount_rur', aggregate_function=sum_jit
            ),
            target_name=f'sum_target_mcc_code_0',
            look_back=1
        )
    ]

    target_mixer = TargetMixerCompose([])

    target_generator_compose = TargetGeneratorCompose(aggregate_target_generators)

    day_boundary_generator = TimeBoundaryGenerator(time_feature='day')
    time_splitting_indices = day_boundary_generator.get_interval_boundaries(data)

    targets = target_generator_compose.generate_targets(
        data, time_splitting_indices
    )

    print('Number of targets', len(targets))
    print('Target shape', targets[
        list(targets.keys())[0]
    ].shape)
    data = {
        "app_id": '0',
        "APPLICATION_DATE": "2044-11-29",
        "feature_arrays": data,
    }

    transforms = Compose([
        Log10ClipShiftTransform(
            field_to_apply='amount_rur',
            field_to_write_transform_result='amount_rur_log10',
            shift=1.
        )
    ])
    numerical_features = [
        ["amount_rur", 1],
        ["amount_rur_log10", 1],
    ]
    numerical_features += [
        [str(i), 1] for i in range(25)
    ]
    dataset = TransactionDatasetWithSyntheticTargetDiskFile(
        data={
            '0': data
        },
        indexes=['0', '1'],
        targets=None,
        length_params={
            'max_days': 500000000, 'lag_days': 1,
            'max_seq_len': length * 2
        },
        features={
            'categorical': [
                ["mcc_code", 3, 3]
            ],
            'numerical': numerical_features
        },
        transforms=transforms,
        augmentations=VCompose([]),
        debug=False,
        dropout=0,
        multi_task=False,
        boundary_generator=day_boundary_generator,
        target_generator=target_generator_compose,
        target_mixer=target_mixer
    )
    indexes = np.concatenate([
        np.full(num_application, fill_value='0')
    ])

    indexes = np.random.permutation(indexes)
    dataset.indexes = indexes
    num_workers = 0
    print('Num workers', num_workers)
    dataloader = DataLoader(dataset, num_workers=num_workers,
                            batch_size=64,
                            collate_fn=SequenceTargetCollatorMultiTarget(seq_len=length),
                            persistent_workers=False)
    print('Init dataloader')
    next(iter(dataloader))

    for batch in tqdm(dataloader):
        pass


if __name__ == '__main__':
    test_pytorch_dataloader_multiple_targets()
