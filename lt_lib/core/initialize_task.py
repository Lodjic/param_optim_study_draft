# Author: Loïc Thiriet

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
from loguru import logger
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lt_lib.core.initialize_model import initialize_model
from lt_lib.data.datasets import generate_dataloaders
from lt_lib.schemas.config_files_schemas import ModelConfig
from lt_lib.schemas.task_schemas import PredictTaskConfig, TaskSchema, TrainTaskConfig
from lt_lib.utils.dict_utils import flatten_dict
from lt_lib.utils.load_and_save import (
    load_pytorch_checkpoint,
    save_dict_as_json,
    save_pytorch_model_checkpoint,
)

######################################
### Parameters extraction function ###
######################################


def extract_task_init_params(task_config: TrainTaskConfig | PredictTaskConfig, model_config: ModelConfig | None):
    init_params_dict = {}

    # Get model initialization params if a model is passed
    if model_config:
        init_params_dict["model_init_params"] = model_config.model_dump(exclude={"config_type", "model"})
        init_params_dict["model_init_params"].update(
            {"model": model_config.model.model_type, "model_parameters": model_config.model.parameters.model_dump()}
        )
    else:
        init_params_dict["model_init_params"] = {}

    # Get dataloaders initialization params
    init_params_dict["dataloaders_init_params"] = task_config.model_dump(
        include={
            "images_extension",
            "batch_size",
            "manual_seed",
        }
    )

    # Get optimizer initialization params
    init_params_dict["optimizer_init_params"] = task_config.model_dump(
        include={
            "optimizer",
            "optimizer_params",
        }
    )

    # Get scheduler
    init_params_dict["lr_scheduler_init_params"] = task_config.model_dump(
        include={
            "lr_scheduler",
            "lr_scheduler_params",
        }
    )

    # Get the task params
    init_params_dict["task_params"] = task_config.model_dump(
        exclude={
            "images_extension",
            "batch_size",
            "optimizer",
            "optimizer_params",
            "lr_scheduler",
            "lr_scheduler_params",
        }
    )

    return init_params_dict


#######################################
### Parameters overriding functions ###
#######################################


def override_saved_model_path(
    inputs_dir: Path, root_outputs_dir: Path, model_input: str, model_init_params: dict[str, Any]
) -> dict[str, Any]:
    model_input = model_input.split(":")
    saved_model_dir = model_input[1]
    model_suffix = model_input[2] if len(model_input) == 3 else "best"

    # If model is saved in the inputs_dir we override the saved_model_pat only if it was not passed in the config
    if saved_model_dir == "inputs_directory":
        if model_init_params["saved_checkpoint_path"] is None:
            model_init_params["saved_checkpoint_path"] = str(list(inputs_dir.glob("*.tar"))[0])
    # Else we suppose it's saved in the output of another task
    else:
        model_init_params["saved_checkpoint_path"] = str(
            list((root_outputs_dir / saved_model_dir).rglob(f"*{model_suffix}.tar"))[0]
        )

    model_init_params["model_loading_type"] = model_input[0]
    model_init_params["weights_backbone"] = None

    return model_init_params


def override_dataframe_path(
    inputs_dir: Path, root_outputs_dir: Path, df_input: str, task_params: dict[str, Any]
) -> dict[str, Path]:
    df_input = df_input.split(":")
    df_type = df_input[0]
    df_dir = df_input[1]

    # We overrride the dataframe path only if it was not passed in the task params
    if task_params[f"{df_type}_path"] is None:

        if df_dir == "inputs_directory":
            if df_type == "gts":
                df_fct = df_input[2] if len(df_input) == 3 else "test"
                task_params[f"{df_type}_path"] = str(
                    list((inputs_dir / f"{df_fct}/annotations").glob(f"*{df_type}*.csv"))[0]
                )
            else:
                task_params[f"{df_type}_path"] = str(list(inputs_dir.glob(f"*{df_type}*.csv"))[0])
        else:
            task_params[f"{df_type}_path"] = str(list((root_outputs_dir / df_dir).glob(f"*{df_type}*.csv"))[0])

    return task_params


def override_config_paths_with_inputs(
    inputs_dir: Path, root_outputs_dir: Path, inputs: str, init_params_dict: dict[str, Any]
):
    for input in inputs.split(","):
        # If 'model' keyword is in the task inputs the saved_checkpoint_path needs to be overriden
        if "model" in input.split(":")[0]:
            init_params_dict["model_init_params"] = override_saved_model_path(
                inputs_dir, root_outputs_dir, input, init_params_dict["model_init_params"]
            )

        # If 'predictions' or 'gts' keyword is in the task inputs the detections_path or gts_path needs to be overriden
        elif input.split(":")[0] in ["predictions", "gts"]:
            init_params_dict["task_params"] = override_dataframe_path(
                inputs_dir, root_outputs_dir, input, init_params_dict["task_params"]
            )

    return init_params_dict


########################################
### Objects initialization functions ###
########################################


def initialize_optimizer(
    model: torch.nn.Module,
    optimizer: str | None = None,
    optimizer_params: dict[str, Any] = {},
    checkpoint_path: Path | None = None,
):
    optimizer_name = optimizer

    if optimizer == "Adam":
        optimizer = Adam(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Parameter optimizer='{optimizer_name}' is not yet implemented!")

    # If a checkpoint_path is passed, then loads it and gets the optimizer_state_dict
    optimizer_resumed = False
    if checkpoint_path:
        optimizer_state_dict = load_pytorch_checkpoint(checkpoint_path)["optimizer_state_dict"]
        # If it's not None, loads state_dict in the optimizer
        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)
            optimizer_resumed = True
            logger.info(
                f"Optimizer '{optimizer_name}' resumed from the state_dict saved in checkpoint file '{checkpoint_path}'."
            )

    if not optimizer_resumed:
        logger.info(f"Optimizer '{optimizer_name}' freshly initiated from scratch.")

    return optimizer, optimizer_resumed


def initialize_lr_scheduler(
    optimizer: str | None = None, lr_scheduler: str | None = None, lr_scheduler_params: dict[str, Any] = {}
):
    if lr_scheduler:
        if lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler = ReduceLROnPlateau(optimizer, **lr_scheduler_params)
        else:
            raise ValueError(f"The `lr_scheduler` value '{lr_scheduler}' has not been implemented yet.")
        return lr_scheduler
    else:
        return None


def initialize_training_objects(
    checkpoint_path: Path | None = None,
    checkpoint_scoring_metric: str | None = None,
    checkpoint_scoring_order: str | None = None,
):
    ORDER_DISPATCHER = {"min": [min, np.inf], "max": [max, 0]}

    # Boolean to know if metrics dict was resumed
    metrics_resumed = False

    # If a checkpoint_path is passed, then loads it
    if checkpoint_path:
        ckpt = load_pytorch_checkpoint(checkpoint_path)
        if ckpt["metrics"]:
            training_objects = {
                "metrics": ckpt["metrics"],
                "epoch": ckpt["epoch"] + 1,  # number of first epoch when resuming
                "checkpoint_scoring_metric_value": (
                    ORDER_DISPATCHER[checkpoint_scoring_order][0](
                        flatten_dict(ckpt["metrics"])[checkpoint_scoring_metric]
                    )
                    if checkpoint_scoring_metric
                    else None
                ),
            }
            metrics_resumed = True
            logger.info(f"Training metrics resumed from the checkpoint file '{checkpoint_path}'.")

    # If metrics were not resumed, then initiates metrics dict from scratch
    if not metrics_resumed:
        metrics = {
            phase: {
                "loss": [],
                "comparable_loss": [],
                "level1": {"tp": [], "fp": [], "fn": [], "recall": [], "precision": []},
                "level2": {"mAP50": [], "mAP": []},
            }
            for phase in ["train", "val"]
        }
        training_objects = {
            "metrics": metrics,
            "epoch": 0,
            "checkpoint_scoring_metric_value": (
                ORDER_DISPATCHER[checkpoint_scoring_order][1] if checkpoint_scoring_order else None
            ),
        }
        logger.info(f"Training metrics freshly initiated from scratch.")

    return training_objects, metrics_resumed


####################################
### Task initialization function ###
####################################


def initialize_task(
    inputs_dir: Path,
    outputs_dir: Path,
    task_schema: TaskSchema,
    model_config: ModelConfig | None = None,
) -> Tuple[dict[str, Any], dict[str, Any]]:

    # Gets the initialization parameters
    init_params_dict = extract_task_init_params(task_schema.config, model_config)
    init_params_dict = override_config_paths_with_inputs(
        inputs_dir, outputs_dir.parent, task_schema.inputs, init_params_dict
    )

    # Initilizes the returned dict with initialized objects and parameters
    initialized_objects = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    initialized_params = {"task_params": init_params_dict["task_params"]}

    # Saves the task initialization parameters for debug
    save_dict_as_json(outputs_dir / f"{task_schema.task_type}_init_params.json", init_params_dict)

    ########## SPECIFIC TO TRAIN AND PREDICT TASKS ##########
    # If the task is of type 'train' or 'predict', instantiates a model
    if task_schema.task_type in ["train", "predict"]:
        if not init_params_dict["model_init_params"]:
            raise ValueError(f"The model config file is missing and needed for a 'train' or 'predict' task! ")

        # Instantiate model
        initialized_objects["model"], saved_checkpoint_path = initialize_model(
            root_data_dir=inputs_dir,
            **init_params_dict["model_init_params"],
        )
        # Sends the model to the correct device
        initialized_objects["model"].to(initialized_objects["device"])

        # Instantiates optimizer if the parameter is passed. This if statement is equivalent to if "train" task.
        if init_params_dict["optimizer_init_params"]:
            initialized_objects["optimizer"], optimizer_resumed = initialize_optimizer(
                model=initialized_objects["model"],
                checkpoint_path=saved_checkpoint_path,
                **init_params_dict["optimizer_init_params"],
            )

            # Instantiates lr_scheduler if one is passed
            if init_params_dict["lr_scheduler_init_params"]:
                initialized_objects["lr_scheduler"] = initialize_lr_scheduler(
                    optimizer=initialized_objects["optimizer"], **init_params_dict["lr_scheduler_init_params"]
                )

            # Instantiates metrics, epoch number and checkpoint_scoring_metric_value
            initialized_objects["training_extra_objects"], metrics_resumed = initialize_training_objects(
                saved_checkpoint_path,
                initialized_params["task_params"]["checkpoint_scoring_metric"],
                initialized_params["task_params"]["checkpoint_scoring_order"],
            )

            # Raise some warning or error if optimizer and metrics are not resumed together
            if metrics_resumed and not optimizer_resumed:
                logger.warning(
                    f"Metrics were resumed from epoch {initialized_objects['training_extra_objects']['epoch']}, but "
                    + "optimizer was not resumed! Are you sure your checkpoint file is not corrupted?"
                )
            if optimizer_resumed and not metrics_resumed:
                logger.error("Optimizer was resumed but not the metrics there must be something wrong.")
                raise ValueError("Optimizer was resumed but not the metrics there must be something wrong.")

            # If the we resume the training we copy the saved model in the outputs as it is the current best model
            if metrics_resumed:
                save_pytorch_model_checkpoint(
                    model=initialized_objects["model"],
                    optimizer=initialized_objects["optimizer"],
                    epoch=initialized_objects["training_extra_objects"]["epoch"],
                    metrics=initialized_objects["training_extra_objects"]["metrics"],
                    model_saving_type=init_params_dict["task_params"]["model_saving_type"],
                    save_optimizer=init_params_dict["task_params"]["save_optimizer"],
                    saving_path=outputs_dir / f"{init_params_dict['task_params']['checkpoint_file_name']}-best.tar",
                )

    # Istantiates dataloaders if dataloaders parameters are passed
    if init_params_dict["dataloaders_init_params"]:
        ds_fct = [input.split(":") for input in task_schema.inputs.split(",") if "dataset" in input][0]
        ds_fct = ds_fct[2] if len(ds_fct) == 3 else "test"
        initialized_objects["dataloaders"] = generate_dataloaders(
            root_data_dir=inputs_dir,
            task_type=task_schema.task_type,
            dataset_fct=ds_fct,
            **init_params_dict["dataloaders_init_params"],
        )

    return initialized_objects, initialized_params
