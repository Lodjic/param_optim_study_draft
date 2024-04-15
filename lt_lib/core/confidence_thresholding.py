from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel

from lt_lib.core.initialize_task import initialize_task
from lt_lib.utils.load_and_save import load_json_as_dict
from lt_lib.utils.log import log_task_begin_or_end
from lt_lib.utils.regex_matcher import get_elements_with_regex


def thresholding_confidence_per_label(
    predictions: pl.DataFrame, threshold_per_label: dict[float, float]
) -> pl.DataFrame:
    # For each class, assigns confidence below threshold to 0 in aliases columns
    for label in threshold_per_label.keys():
        predictions = predictions.with_columns(
            pl.when(pl.col(f"^.*confidence.*{label}.*$") < threshold_per_label[label])
            .then(0)
            .otherwise(pl.col(f"^.*confidence.*{label}.*$"))
            .alias(
                f"{get_elements_with_regex(f'^.*confidence.*{label}.*$', predictions.columns, unique=True)}_for_filtering"
            )
        )

    # Filters predictions that are below threshold for all classes
    predictions = predictions.filter(pl.all_horizontal(pl.col(f"^.*confidence.*_for_filtering.*$") == 0).not_())

    # Get columns names of the duplicated confidence columns for filtering
    confidence_for_filtering_columns = get_elements_with_regex(
        f"^.*confidence.*_for_filtering.*$", predictions.columns, unique=False
    )

    # Saves the max confidence value of the kept columns
    predictions = predictions.with_columns(max_confidence=pl.max_horizontal(pl.col("^.*confidence.*_for_filtering.*$")))

    # Removes columns used for filtering
    predictions = predictions.drop(confidence_for_filtering_columns)

    # Finds the most probable label according to max_confidence saved
    predictions_probable_label = pl.DataFrame()
    for label in threshold_per_label.keys():
        predictions_probable_label_temp = predictions.filter(
            pl.col(f"^.*confidence.*{label}.*$") == pl.col("max_confidence")
        ).select(["id"])
        predictions_probable_label_temp = predictions_probable_label_temp.with_columns(
            probable_label=pl.lit(int(label))
        )
        predictions_probable_label = pl.concat(
            [predictions_probable_label, predictions_probable_label_temp], how="vertical"
        )

    # Remove probable_label and max_confidence columns
    predictions = predictions.drop(["probable_label", "max_confidence"])

    # Adds the newly computed probable labels
    predictions = predictions.join(predictions_probable_label, on="id")

    return predictions


def thresholding_condifdence_per_label_np(
    predictions: pl.DataFrame, threshold_per_label: dict[float, float]
) -> pl.DataFrame:
    # For each class, assigns confidence below threshold to 0
    for label in threshold_per_label.keys():
        predictions = predictions.with_columns(
            pl.when(pl.col(f"^.*confidence.*{label}.*$") < threshold_per_label[label])
            .then(0)
            .otherwise(pl.col(f"^.*confidence.*{label}.*$"))
            .alias(
                f"{get_elements_with_regex(f'^.*confidence.*{label}.*$', predictions.columns, unique=True)}_for_filtering"
            )
        )

    # Filters predictions that are below threshold for all classes
    predictions = predictions.filter(pl.all_horizontal(pl.col(f"^.*confidence.*_for_filtering.*$") == 0).not_())

    # Get columns names of the duplicated confidence columns for filtering
    confidence_for_filtering_columns = get_elements_with_regex(
        f"^.*confidence.*_for_filtering.*$", predictions.columns, unique=False
    )

    # Finds the most probable label according to max_confidence on confidence_for_filtering columns
    probable_labels = predictions.select(confidence_for_filtering_columns).to_numpy().argmax(axis=1) + 1

    # Removes columns used for filtering
    predictions = predictions.drop(confidence_for_filtering_columns)

    # Adds the newly computed probable labels
    predictions = predictions.with_columns(probable_label=probable_labels)

    return predictions


def thresholding_predictions_confidence(
    predictions_path: str | Path,
    threshold: dict[float, float] | dict[str, float] | float,
    gts_path: str | Path | None,
    outputs_dir: Path,
):
    # Load predictions csv files
    predictions = pl.read_csv(predictions_path)

    # If there are no predictions, it's not necessary to run any thresholding
    if len(predictions) > 0:
        if isinstance(threshold, float):
            threshold_per_label = {label: threshold for label in predictions["probable_label"].unique()}
        elif isinstance(list(threshold.keys())[0], str):
            if gts_path is None:
                raise ValueError(
                    f"The parameter `gts_path` should be passed to the thresholding task or filled in the inputs because "
                    + "labels aassociated with label_names cannot be infered"
                )
            else:
                label_to_label_name = load_json_as_dict(Path(gts_path).parent / "label_to_label_name.json")
                label_name_to_label = {v: k for (k, v) in label_to_label_name.items()}
                threshold_per_label = {
                    label_name_to_label[label_name]: threshold[label_name] for label_name in threshold.keys()
                }
        else:
            threshold_per_label = threshold

        # Apply threshold per label on all predictions
        predictions = thresholding_confidence_per_label(predictions, threshold_per_label)

    # Saves thresholded predictions
    predictions.write_csv(outputs_dir / "predictions.csv", include_header=True)


def confidence_thresholding(
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
    log_task_begin_or_end("confidence_thresholding", "begin", "lower")

    # Launch prediction
    thresholding_predictions_confidence(
        outputs_dir=outputs_dir,
        **initialized_params["task_params"],
    )

    # Log end of eval
    log_task_begin_or_end("confidence_thresholding", "end", "lower")
