import pickle
from typing import Any, Dict, Optional, Iterable, List

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import hydra

from src.data.components.collate import Collator
from src.data.components.data_reader import DataReader
from src.data.components.dataset import TransactionDatasetDiskFile, DummyDataset
from src.data.components.target_reader import TargetReader, IndexReader
from src.data.components.transforms import Compose
from src.data.components.feature_selector import FeatureSelector
from src.data.components.augmentation import VCompose


import numpy as np
from src import utils
# from src.utils.sampler import BalanceClassSampler, SamplerMultiTarget

log = utils.get_pylogger(__name__)


class TransactionDataModuleDiskFile(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
            self,
            data: DataReader,
            indexes_reader: IndexReader,
            targets_reader: TargetReader,
            features=None,
            length_params=None,
            transforms: Compose = None,
            feature_selector: FeatureSelector = None,
            augmentation: VCompose = None,
            collator=None,
            train_batch_size: int = 64,
            train_balance_sampler=False,
            val_batch_size: int = 64,
            num_workers: int = 0,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False,
            debug=False,
            dataset_dropout=0.,
            drop_last: bool = False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data: DataReader = hydra.utils.instantiate(data)
        self.targets_reader: TargetReader = hydra.utils.instantiate(targets_reader)
        self.indexes_reader: IndexReader = hydra.utils.instantiate(indexes_reader)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.train_sampler = None

        self.data_predict: Optional[Dataset] = None
        self.predict_dataset_name: Optional[Dataset] = None

        self.data_hashes: Optional[Dict[str, str]] = None
        self.target_hash: Optional[str] = None
        self.index_hashes: Optional[Dict[str, str]] = None

    @property
    def num_classes(self):
        return 10

    def set_predict_data(self, dataset_name):
        self.predict_dataset_name = dataset_name

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def _get_hashes(self):
        self.target_hash = self.targets_reader.hash_sum
        self.data_hashes = self.data.hash_sum
        self.index_hashes = {
            'train_indexes': self.indexes_reader.train_hash,
            'val_indexes': self.indexes_reader.val_hash,
            'test_indexes': self.indexes_reader.test_hash,
            'predict_indexes': self.indexes_reader.predict_hash
        }

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        log.info(f"Setup transforms ...")
        transforms: Compose = hydra.utils.instantiate(self.hparams.transforms)
        transforms.setup()
        log.info("Setup feature_selector")
        features = self.hparams.features
        if self.hparams.feature_selector:
            feature_selector: FeatureSelector = hydra.utils.instantiate(self.hparams.feature_selector)
            feature_selector.setup(features)
            features = feature_selector.get_output_features()

        log.info(f"Setuo augmentation ...")
        augmentation: VCompose = hydra.utils.instantiate(self.hparams.augmentations)
        log.info(f"Setup data ...")
        self.data.setup()
        log.info(f"Setup targets ...")
        self.targets_reader.setup()
        log.info(f"Setup train dataset ...")
        self.data_train = TransactionDatasetDiskFile(
            data=self.data,
            indexes=self.indexes_reader.train_indexes,
            targets=self.targets_reader.targets,
            length_params=self.hparams.length_params,
            features=features,
            transforms=transforms,
            debug=self.hparams.debug,
            dropout=self.hparams.dataset_dropout
        )
        if self.hparams.train_balance_sampler:
            # train_targets = [self.data_train.targets[idx] for idx in self.data_train.indexes]
            # train_targets = np.array(train_targets)
            # log.info(f"Tgt shape... {train_targets.shape}")
            #
            # self.train_sampler = BalanceClassSampler(
            #     labels=train_targets, mode='downsampling'
            # )
            pass

        log.info(f"Setup val dataset ...")
        self.data_val = TransactionDatasetDiskFile(
            data=self.data,
            indexes=self.indexes_reader.val_indexes,
            targets=self.targets_reader.targets,
            length_params=self.hparams.length_params,
            features=features,
            transforms=transforms,
            feature_selector=feature_selector,
            debug=self.hparams.debug
        )
        self.data_test = self.data_val
        self._set_predict_loader()
        self._get_hashes()

    def _set_predict_loader(self):
        if self.predict_dataset_name:
            if self.predict_dataset_name == 'train':
                self.data_predict = self.data_train
            elif self.predict_dataset_name == 'val':
                self.data_predict = self.data_val
            elif self.predict_dataset_name == 'test':
                self.data_predict = self.data_test
            else:
                raise ValueError('Dataset should be train/val/test')

    def train_dataloader(self):
        collate_fn: Collator = hydra.utils.instantiate(self.hparams.collator)
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.train_sampler is None,
            collate_fn=collate_fn,
            sampler=self.train_sampler,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
            drop_last=self.hparams.drop_last
        )

    def val_dataloader(self):
        collate_fn: Collator = hydra.utils.instantiate(self.hparams.collator)
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers
        )

    def test_dataloader(self):
        collate_fn: Collator = hydra.utils.instantiate(self.hparams.collator)
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.val_batch_size,  # add test_batch_size
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )

    def predict_dataloader(self):
        collate_fn: Collator = hydra.utils.instantiate(self.hparams.collator)
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


class TransactionDataModuleDiskFileDEBUG(TransactionDataModuleDiskFile):
    def __init__(
            self,
            debug_samples_path, *args, **kwargs,

    ):
        super().__init__(*args, **kwargs, )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        # log.info(f"Setup transforms ...")
        # transforms: Compose = hydra.utils.instantiate(self.hparams.transforms)
        # transforms.setup()
        # log.info("Setup feature_selector")
        # features = self.hparams.features
        # if self.hparams.feature_selector:
        #     feature_selector: FeatureSelector = hydra.utils.instantiate(self.hparams.feature_selector)
        #     feature_selector.setup(features)
        #     features = feature_selector.get_output_features()
        # feature_selector.setup()
        # log.info(f"Setup data ...")
        # self.data.setup()
        # log.info(f"Setup targets ...")
        # self.targets_reader.setup()
        # log.info(f"Setup train dataset ...")
        with open(self.hparams.debug_samples_path, 'rb') as f:
            samples = pickle.load(f)
        self.data_train = DummyDataset(samples)

        log.info(f"Setup val dataset ...")

        self.data_val = DummyDataset(samples)

        if self.predict_dataset_name:
            if self.predict_dataset_name == 'train':
                self.data_predict = self.data_train
            elif self.predict_dataset_name == 'val':
                self.data_predict = self.data_val
            elif self.predict_dataset_name == 'test':
                self.data_predict = self.data_test
            else:
                raise ValueError('Dataset should be train/val/test')

        self._set_predict_loader()
        self._get_hashes_fake()

    def _get_hashes_fake(self):
        self.target_hash = 'fake_hash'
        self.data_hashes = {'debug_samples': 'fake_hash'}
        self.index_hashes = {
            'train_indexes': 'fake_hash',
            'val_indexes': 'fake_hash',
            'test_indexes': 'fake_hash',
            'predict_indexes': 'fake_hash',
        }


class TransactionDataModuleDiskFileMultival(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
            self,
            data: DataReader,
            indexes_reader: IndexReader,
            targets_reader: TargetReader,
            features=None,
            length_params=None,
            transforms: Compose = None,
            feature_selector: FeatureSelector = None,
            augmentations: VCompose = None,
            collator=None,
            train_batch_size: int = 64,
            train_balance_sampler=False,
            val_batch_size: int = 64,
            num_workers: int = 0,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False,
            debug=False,
            dataset_dropout=0.,
            multi_task=None,
            n_samples=None
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data: DataReader = hydra.utils.instantiate(data)
        self.targets_reader: TargetReader = hydra.utils.instantiate(targets_reader)
        self.indexes_reader: IndexReader = hydra.utils.instantiate(indexes_reader)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[List[Dataset]] = None
        self.data_test: Optional[List[Dataset]] = None
        self.train_sampler = None

        self.data_predict: Optional[Dataset] = None
        self.predict_dataset_name: Optional[Dataset] = None

        self.data_hashes: Optional[Dict[str, str]] = None
        self.target_hash: Optional[str] = None
        self.index_hashes: Optional[Dict[str, str]] = None
        self.multi_task = multi_task
        self.n_samples = n_samples

    @property
    def num_classes(self):
        return 10

    def set_predict_data(self, dataset_name):
        self.predict_dataset_name = dataset_name

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def _get_hashes(self):
        self.target_hash = self.targets_reader.hash_sum
        self.data_hashes = self.data.hash_sum
        self.index_hashes = {
            'train_indexes': self.indexes_reader.train_hash,
            'val_indexes': self.indexes_reader.val_hash,
            'test_indexes': self.indexes_reader.test_hash,
            'predict_indexes': self.indexes_reader.predict_hash
        }

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        setup_data = lambda indexes, dropout=0.0: TransactionDatasetDiskFile(
            data=self.data,
            indexes=indexes,
            targets=self.targets_reader.targets,
            length_params=self.hparams.length_params,
            features=features,
            transforms=transforms,
            augmentations=augmentations,
            debug=self.hparams.debug,
            dropout=dropout,
            multi_task=self.multi_task
        )

        log.info(f"Setup transforms ...")
        transforms: Compose = hydra.utils.instantiate(self.hparams.transforms)
        transforms.setup()
        log.info("Setup feature_selector")
        features = self.hparams.features
        if self.hparams.feature_selector:
            feature_selector: FeatureSelector = hydra.utils.instantiate(self.hparams.feature_selector)
            feature_selector.setup(features)
            features = feature_selector.get_output_features()

        log.info(f"Setup augmentation ...")
        augmentations: VCompose = hydra.utils.instantiate(self.hparams.augmentations)
        log.info(f"Setup data ...")
        self.data.setup()
        log.info(f"Setup targets ...")
        self.targets_reader.setup()

        if isinstance(self.indexes_reader.train_path, str):
            log.info(f"Setup train dataset ...")
            self.data_train = setup_data(self.indexes_reader.train_indexes, self.hparams.dataset_dropout)
        else:
            log.info(f"Setup train datasets ...")
            self.data_train = [setup_data(indexes, self.hparams.dataset_dropout) for indexes in
                               self.indexes_reader.train_indexes]

        if self.hparams.train_balance_sampler:
            # if self.multi_task:
            #     self.train_sampler = SamplerMultiTarget(
            #         indexes_len=len(self.indexes_reader.train_indexes), n_samples=self.n_samples
            #     )
            # elif not isinstance(self.data_train, Iterable):
            #     train_targets = [self.data_train.targets[idx] for idx in self.data_train.indexes]
            #     train_targets = np.array(train_targets)
            #     log.info(f"Tgt shape... {train_targets.shape}")
            #
            #     self.train_sampler = BalanceClassSampler(
            #         labels=train_targets, mode='downsampling'
            #     )
            # else:
            #     self.train_sampler = list()
            #     for dataset_idx, dataset in enumerate(self.data_train):
            #         train_targets = [self.data_train[dataset_idx].targets[idx] for idx in
            #                          self.data_train[dataset_idx].indexes]
            #         train_targets = np.array(train_targets)
            #         log.info(f"Tgt_{dataset_idx} shape... {train_targets.shape}")
            #
            #         self.train_sampler.append(BalanceClassSampler(
            #             labels=train_targets, mode='downsampling'
            #         ))
            pass

        if isinstance(self.indexes_reader.val_path, str):
            log.info(f"Setup val dataset ...")
            self.data_val = setup_data(self.indexes_reader.val_indexes)
        else:
            log.info(f"Setup val datasets ...")
            self.data_val = [setup_data(indexes) for indexes in self.indexes_reader.val_indexes]
        self.data_test = self.data_val
        self._set_predict_loader()
        self._get_hashes()

    def _set_predict_loader(self):
        if self.predict_dataset_name:
            if self.predict_dataset_name == 'train':
                self.data_predict = self.data_train
            elif self.predict_dataset_name == 'val':
                self.data_predict = self.data_val
            elif self.predict_dataset_name == 'test':
                self.data_predict = self.data_test
            else:
                raise ValueError('Dataset should be train/val/test')

    def train_dataloader(self):
        collate_fn: Collator = hydra.utils.instantiate(self.hparams.collator)
        setup_dataloader = lambda dataset, train_sampler: DataLoader(
            dataset=dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=train_sampler is None,
            collate_fn=collate_fn,
            sampler=train_sampler,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers
        )
        if isinstance(self.data_train, Iterable):
            if self.train_sampler is None:
                return [setup_dataloader(dataset, None) for dataset in self.data_train]
            else:
                # return [setup_dataloader(dataset, train_sampler) for dataset, train_sampler in
                #         zip(self.data_train, self.train_sampler)]
                pass
        else:
            return setup_dataloader(self.data_train, self.train_sampler)

    def val_dataloader(self):
        collate_fn: Collator = hydra.utils.instantiate(self.hparams.collator)
        setup_dataloader = lambda dataset: DataLoader(
            dataset=dataset,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers
        )
        if isinstance(self.data_val, Iterable):
            return [setup_dataloader(dataset) for dataset in self.data_val]
        else:
            return setup_dataloader(self.data_val)

    def test_dataloader(self):
        collate_fn: Collator = hydra.utils.instantiate(self.hparams.collator)
        setup_dataloader = lambda dataset: DataLoader(
            dataset=dataset,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )
        if isinstance(self.data_test, Iterable):
            return [setup_dataloader(dataset) for dataset in self.data_test]
        else:
            return setup_dataloader(self.data_test)

    def predict_dataloader(self):
        collate_fn: Collator = hydra.utils.instantiate(self.hparams.collator)
        setup_dataloader = lambda dataset: DataLoader(
            dataset=dataset,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn
        )
        if isinstance(self.data_predict, Iterable):
            return [setup_dataloader(dataset) for dataset in self.data_predict]
        else:
            return setup_dataloader(self.data_predict)

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


class TransactionDataModuleDiskFileMultivalDEBUG(TransactionDataModuleDiskFileMultival):
    def __init__(
            self,
            debug_samples_path, *args, **kwargs,

    ):
        super().__init__(*args, **kwargs, )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        # log.info(f"Setup transforms ...")
        # transforms: Compose = hydra.utils.instantiate(self.hparams.transforms)
        # transforms.setup()
        # log.info("Setup feature_selector")
        # features = self.hparams.features
        # if self.hparams.feature_selector:
        #     feature_selector: FeatureSelector = hydra.utils.instantiate(self.hparams.feature_selector)
        #     feature_selector.setup(features)
        #     features = feature_selector.get_output_features()
        # log.info(f"Setup data ...")
        # self.data.setup()
        # log.info(f"Setup targets ...")
        # self.targets_reader.setup()
        # log.info(f"Setup train dataset ...")
        with open(self.hparams.debug_samples_path, 'rb') as f:
            samples = pickle.load(f)
        self.data_train = DummyDataset(samples)

        log.info(f"Setup val dataset ...")

        if isinstance(self.indexes_reader.val_path, str):
            log.info(f"Setup val dataset ...")
            self.data_val = DummyDataset(samples)
        else:
            log.info(f"Setup val datasets ...")
            self.data_val = [DummyDataset(samples) for indexes in self.indexes_reader.val_indexes]

        if self.predict_dataset_name:
            if self.predict_dataset_name == 'train':
                self.data_predict = self.data_train
            elif self.predict_dataset_name == 'val':
                self.data_predict = self.data_val
            elif self.predict_dataset_name == 'test':
                self.data_predict = self.data_test
            else:
                raise ValueError('Dataset should be train/val/test')

        self._set_predict_loader()
        self._get_hashes_fake()

    def _get_hashes_fake(self):
        self.target_hash = 'fake_hash'
        self.data_hashes = {'debug_samples': 'fake_hash'}
        self.index_hashes = {
            'train_indexes': 'fake_hash',
            'val_indexes': 'fake_hash',
            'test_indexes': 'fake_hash',
            'predict_indexes': 'fake_hash',
        }


if __name__ == "__main__":
    _ = TransactionDataModuleDiskFile()
