import pickle
from dataclasses import asdict

import hydra
import numpy as np
import os
import rootutils
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# log = RankedLogger(__name__, rank_zero_only=True)

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force app to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #
# pylint: disable=wrong-import-position
from src.dataset_generate.src.target import TargetGenerator
from src.dataset_generate.src.transaction import BaseAppTransactionSampler


def instantiate_omega_conf_resolvers():
    """Add to omega conf resolvers for custom functions."""
    OmegaConf.register_new_resolver("div", lambda x, y: x // y)
    OmegaConf.register_new_resolver("div_perc", lambda x, y: int(x) % int(y))
    OmegaConf.register_new_resolver("get_folder_name", lambda x: x.split("/")[-1])
    OmegaConf.register_new_resolver(
        "split_and_get_by_index", lambda s, delimiter, index: s.split(delimiter)[index]
    )
    OmegaConf.register_new_resolver("mult", lambda x, y: x * y)
    OmegaConf.register_new_resolver(
        "log_exp", lambda x: round(float(np.log(x)), 3)
    )
    OmegaConf.register_new_resolver("exp", lambda x: round(float(np.exp(x)), 3))
    OmegaConf.register_new_resolver("plus", lambda x, y: x + y)
    OmegaConf.register_new_resolver("minus", lambda x, y: x - y)
    OmegaConf.register_new_resolver("__getitem__", lambda x, idx: x[idx])


instantiate_omega_conf_resolvers()


@hydra.main(version_base="1.3", config_path="configs", config_name="generate.yaml")
def main(cfg: DictConfig):
    """Main entry point for dataset generation.

    Args:
        cfg (DictConfig): configuration composed by Hydra.
    """

    app_sampler: BaseAppTransactionSampler = hydra.utils.instantiate(
        cfg.transaction_sampler.app_sampler
    )
    target_generator: TargetGenerator = hydra.utils.instantiate(cfg.target_generator)

    train_val_test_parts = [cfg.train_part, cfg.callback_part, cfg.val_part, cfg.test_part]
    train_val_test_parts = [int(part * cfg.num_applications) for part in train_val_test_parts]

    assert (
        sum(train_val_test_parts) == cfg.num_applications
    ), "sum of train, val and test should be equals to size of dataset"

    apps = []
    targets = {}
    indexes = []
    for app_num in tqdm(range(cfg.num_applications)):
        # sample app transactions
        app = app_sampler.sample()

        # calculate target for app
        target = target_generator.get_target(app)

        # for serialization
        app.mcc_code = app.mcc_code.numpy()
        app.amount_rur = app.amount_rur.numpy()
        if app.days_before_application is not None:
            app.days_before_application = app.days_before_application.numpy()

        app_id = str(app_num)
        # APPLICATION_DATE
        apps.append(
            {
                "app_id": app_id,
                "APPLICATION_DATE": "2021-11-29",  # может сломаться?
                "feature_arrays": asdict(app),
            }
        )
        apps[-1]["feature_arrays"]["local_date"] = np.array(
            [np.datetime64("2021-11-12T13:21:52.000000")] * len(app.mcc_code)
        )  # вынести в генератор
        apps[-1]["feature_arrays"]["trans_type"] = np.random.choice(
            [1, 3, 4, 7, 9], len(app.mcc_code)
        )
        apps[-1]["feature_arrays"]["trans_country"] = np.random.choice(
            [15, 217], len(app.mcc_code)
        )
        apps[-1]["feature_arrays"]["trans_curency"] = np.random.choice([110], len(app.mcc_code))

        targets[app_id] = target
        indexes.append(app_id)
    transaction_path = os.path.join(cfg.save_path, 'transactions.pickle')
    os.makedirs(cfg.save_path, exist_ok=True)

    with open(transaction_path, 'wb') as f:
        pickle.dump(apps, f)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
