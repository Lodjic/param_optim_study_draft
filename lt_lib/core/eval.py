from functools import partial
from multiprocessing import Pool
from pathlib import Path

import polars as pl
import ray.train as ray_train
import wandb
from loguru import logger
from pydantic import BaseModel
from tqdm.auto import tqdm

from lt_lib.core.initialize_task import initialize_task
from lt_lib.core.matching import (
    filter_ids_with_low_iou,
    match_predictions_and_gts_for_one_img,
)
from lt_lib.core.metrics import *
from lt_lib.data.datasets import POLARS_GTS_SCHEMA
from lt_lib.utils.dict_utils import flatten_dict
from lt_lib.utils.load_and_save import load_json_as_dict, save_dict_as_json
from lt_lib.utils.log import log_task_begin_or_end
from lt_lib.utils.regex_matcher import get_elements_with_regex


def match_and_compute_eval_one_img(
    img_name: str | Path,
    gts: pl.DataFrame,
    predictions: pl.DataFrame,
    label_to_label_name_dict: dict[str, str],
    matching_iou_threshold: float = 0.4,
):
    # Filter image gts and predictions
    img_gts = gts.filter(pl.col("img_name") == img_name)
    img_predictions = predictions.filter(pl.col("img_name") == img_name)

    # Get image gts and predictions bboxes
    img_gts_bboxes = img_gts[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].to_numpy()
    img_predictions_bboxes = img_predictions[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].to_numpy()

    # Computes matching
    img_predictions_matched_idx, img_gts_matched_idx, img_matched_ious = match_predictions_and_gts_for_one_img(
        img_predictions_bboxes,
        img_gts_bboxes,
    )
    # Filters low iou matches
    img_predictions_matched_idx, img_gts_matched_idx, _ = filter_ids_with_low_iou(
        img_predictions_matched_idx, img_gts_matched_idx, img_matched_ious, matching_iou_threshold
    )

    # Compute img metrics
    img_metrics = compute_img_metrics(
        img_gts, img_predictions, img_gts_matched_idx, img_predictions_matched_idx, label_to_label_name_dict
    )
    return [img_name, img_metrics]


def match_and_compute_eval(
    predictions_path: str | Path,
    gts_path: str | Path,
    outputs_dir: Path,
    matching_iou_threshold: float = 0.4,
    custom_metrics: dict[str, str] | None = None,
    disable_tqdm: bool | None = None,
):
    # Load gts and predictions csv files
    predictions = pl.read_csv(predictions_path)
    gts = pl.read_csv(gts_path).cast(POLARS_GTS_SCHEMA, strict=True)

    label_to_label_name_dict = load_json_as_dict(Path(gts_path).parent / "label_to_label_name.json")

    # Get the image names list
    img_name_list = gts["img_name"].unique().to_list()

    # Initiate results metrics dict
    metrics = {"global_results": {}, "per_img_results": {}}

    # Processes images 1 by 1
    for img_name in tqdm(img_name_list, disable=disable_tqdm):
        _, img_metrics = match_and_compute_eval_one_img(
            img_name, gts, predictions, label_to_label_name_dict, matching_iou_threshold
        )
        metrics["per_img_results"][img_name] = img_metrics

    # Computes global metrics
    global_results = compute_global_results(metrics["per_img_results"])

    # Adds custom metrics if there are some
    if custom_metrics:
        global_results = add_custom_metrics(global_results, custom_metrics)

    # Saves global_metrics to metrics dict
    metrics["global_results"] = global_results

    # Save eval locally
    save_dict_as_json(file_path=outputs_dir / "eval.json", dictionary=metrics)

    ### Ray and wandb logging

    # Flattens global_metrics dict for reporting if necessary
    if ray_train.context.session.get_session() or wandb.run:
        flattened_global_results = flatten_dict(metrics["global_results"])

    # If wandb is running, then logs interesting metrics
    if wandb.run:
        metrics_to_keep = get_elements_with_regex(
            "^(level1.recall|level1.precision|.*civil.*\.recall.*|.*civil.*\.precision.*).*$",
            flattened_global_results.keys(),
            unique=False,
        )
        wandb.log({m: flattened_global_results[m] for m in metrics_to_keep})

    # If Ray-tune is operating, then logs the metrics to it
    if ray_train.context.session.get_session():
        ray_train.report(flattened_global_results)


def match_and_compute_eval_multi_process(
    predictions_path: str | Path,
    gts_path: str | Path,
    outputs_dir: Path,
    matching_iou_threshold: float = 0.4,
    processes: int = 6,
    chunksize: int = 20,
    disable_tqdm: bool | None = None,
):
    # Load gts and predictions csv files
    predictions = pl.read_csv(predictions_path)
    gts = pl.read_csv(gts_path).cast(POLARS_GTS_SCHEMA, strict=True)

    label_to_label_name_dict = load_json_as_dict(Path(gts_path).parent / "label_to_label_name.json")

    # Get the image names list
    img_name_list = gts["img_name"].unique().to_list()

    # Initiate results metrics dict
    metrics = {"global_results": {}, "per_img_results": {}}

    # Distribute files accross processes file by file
    with Pool(processes=processes) as pool:
        with tqdm(total=len(img_name_list), disable=disable_tqdm) as pbar:
            for img_metrics in pool.imap(
                partial(
                    match_and_compute_eval_one_img,
                    gts=gts,
                    predictions=predictions,
                    label_to_label_name_dict=label_to_label_name_dict,
                    matching_iou_threshold=matching_iou_threshold,
                ),
                img_name_list,
                chunksize=chunksize,
            ):
                metrics["per_img_results"][img_metrics[0]] = img_metrics[1]
                pbar.update()

    # Compute global metrics
    metrics["global_results"] = compute_global_results(metrics["per_img_results"])

    save_dict_as_json(file_path=outputs_dir / "eval.json", dictionary=metrics)


def evaluation(
    inputs_dir: Path,
    outputs_dir: Path,
    configs: dict[str, BaseModel],
):
    # Initialize model and dataloader
    _, initialized_params = initialize_task(
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        task_schema=configs["task_schema"],
    )

    # Log end of eval
    log_task_begin_or_end("eval", "begin", "lower")

    # Launch prediction
    match_and_compute_eval(
        outputs_dir=outputs_dir,
        **initialized_params["task_params"],
    )

    # Log end of eval
    log_task_begin_or_end("eval", "end", "lower")
