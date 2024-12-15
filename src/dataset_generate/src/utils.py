from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from importlib.util import find_spec
from typing import Any, Dict, List, Tuple

import torch
from omegaconf import DictConfig, OmegaConf


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt app to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: str | None) -> float | None:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise ValueError(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


#
def cast_to_tensor_or_keep_list(elements: List[float | int | torch.Tensor]) -> List[torch.Tensor]:
    """Convert to list of tensors."""
    if isinstance(elements[0], torch.Tensor):
        return elements

    return [torch.tensor([element]) for element in elements]


def merge_dataclasses_with_torch_tensors(objects: List[Any]) -> dict:
    """Merge list of dataclasses into one with concatenated objects in all fields
    Args:
        objects (Any): list of the same dataclass instances
    Returns:
        Dict with concatenated all fields of all dataclasses in objects list
    """
    first_object = objects[0]
    if not is_dataclass(first_object):
        raise ValueError("Passed values should be dataclasses")

    field_names = [f.name for f in fields(first_object)]
    result_dict = {name: [] for name in field_names}
    for obj in objects:
        for name in field_names:
            result_dict[name].append(getattr(obj, name))
    result_dict = {k: torch.concat(cast_to_tensor_or_keep_list(v)) for k, v in result_dict.items()}
    return result_dict


def instantiate_omega_conf_resolvers():
    """Add to omega conf resolvers for custom functions."""
    OmegaConf.register_new_resolver("div", lambda x, y: x // y)
    OmegaConf.register_new_resolver("div_perc", lambda x, y: int(x) % int(y))
    OmegaConf.register_new_resolver("get_folder_name", lambda x: x.split("/")[-1])
    OmegaConf.register_new_resolver(
        "split_and_get_by_index", lambda s, delimiter, index: s.split(delimiter)[index]
    )
    OmegaConf.register_new_resolver("mult", lambda x, y: x * y)


def flatten_dict(input_dict, parent_key="", sep="."):
    """Flatten a nested dictionary with keys merged by a separator.

    Args:
    - input_dict (dict): The nested dictionary to be flattened.
    - parent_key (str): Internal parameter for recursion, leave empty when calling.
    - sep (str): Separator to merge keys.

    Returns:
    - dict: Flattened dictionary.
    """
    items = []
    for key, value in input_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        elif isinstance(value, list):
            for i, v in enumerate(value):
                list_key = f"{new_key}{sep}{i}"
                if isinstance(v, (dict, list)):
                    items.extend(flatten_dict({list_key: v}, sep=sep).items())
                else:
                    items.append((list_key, v))
        else:
            items.append((new_key, value))
    return dict(items)
