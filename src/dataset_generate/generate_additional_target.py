import cProfile
import os
import pickle  # nosec
from typing import Union

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


rootpath = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# pylint: disable=wrong-import-position
from src.dataset_generate.src.target import (
    SlidingWindowTargetGenerator,
    TargetGenerator,
)
from src.dataset_generate.src.transaction import AppFeaturesBase


@hydra.main(
    version_base="1.3", config_path="configs", config_name="generate_additional_target.yaml"
)
def main(cfg: DictConfig):
    """Main entry point for generation target for dataset.

    Args:
        cfg (DictConfig): configuration composed by Hydra.
    """
    target_generator: Union[TargetGenerator, SlidingWindowTargetGenerator] = (
        hydra.utils.instantiate(cfg.target_generator)
    )
    with open(os.path.join(cfg.dataset_folder, 'transactions.pickle'), 'rb') as f:
        apps = pickle.load(f)

    def generate(apps, target_generator):
        targets = []
        for app in tqdm(apps):
            app_features = AppFeaturesBase.from_feature_array(app["feature_arrays"])
            target = target_generator.get_target(app_features)
            targets.append(target)
        return targets

    cProfile.runctx('generate(apps, target_generator)', None, locals(), filename='base_target_generation.pstat')


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
