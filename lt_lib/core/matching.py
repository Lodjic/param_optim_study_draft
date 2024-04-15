# Author : Loïc Thiriet

from typing import Literal

import numpy as np
import polars as pl
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torchvision.ops import boxes as box_ops

from lt_lib.core.predict import export_predictions_to_df

POLARS_AP_DF_SCHEMA = {
    "id": pl.UInt32,
    "img_name": pl.Utf8,  # pl.Utf8 because Colab does not have pl.String
    "confidence": pl.Float32,
    "label": pl.UInt8,
    "iou": pl.Float32,
    "match": pl.UInt8,
    "correct_match": pl.UInt8,
}


def _upcast(t: np.ndarray) -> np.ndarray:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if np.issubdtype(type(t), np.floating):
        return t if t.dtype in (np.float32, np.float64) else t.astype(np.float64)
    else:
        return t if t.dtype in (np.int32, np.int64) else t.astype(np.int64)


def box_area(boxes: np.ndarray) -> np.ndarray:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (np.ndarray[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        np.ndarray[N]: the area for each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def boxes_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (np.ndarray[N, 4]): first set of boxes
        boxes2 (np.ndarray[M, 4]): second set of boxes

    Returns:
        np.ndarray[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = np.clip(_upcast(rb - lt), a_min=0, a_max=None)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter

    iou = inter / union

    return iou


def match_predictions_and_gts_for_one_img(
    img_predictions_bboxes: np.ndarray | Tensor,
    img_gts_bboxes: np.ndarray | Tensor,
) -> tuple[np.ndarray[int], np.ndarray[int], np.ndarray[float]]:
    # Computes the iou matrix between detecions and gts
    if torch.is_tensor(img_predictions_bboxes):
        iou_matrix = box_ops.box_iou(img_predictions_bboxes, img_gts_bboxes)
        iou_matrix = iou_matrix.detach().cpu().numpy()
    else:
        iou_matrix = boxes_iou(img_predictions_bboxes, img_gts_bboxes)

    # Hugarian algorithm to match predictions with gts. It returns predictions_indexes and column_indexes matched
    predictions_idx, gts_idx = linear_sum_assignment(1 - iou_matrix)

    return predictions_idx, gts_idx, iou_matrix[predictions_idx, gts_idx]


def filter_ids_with_low_iou(
    predictions_idx: np.ndarray[int],
    gts_idx: np.ndarray[int],
    ious: np.ndarray[float],
    matching_iou_threshold: float = 0.4,
):
    # Creates the filter mask to remove low iou matching
    sufficient_iou_mask = ious >= matching_iou_threshold

    return predictions_idx[sufficient_iou_mask], gts_idx[sufficient_iou_mask], ious[sufficient_iou_mask]


def batch_matching_during_training(
    imgs_path: list[str],
    batch_predictions: list[dict[str, np.ndarray | Tensor]],
    batch_gts: list[dict[str, np.ndarray | Tensor]],
    with_bboxes: bool = False,
    confidence_columns: Literal["all", "max"] = "max",
    label_column_name: str = "label",
) -> pl.DataFrame:
    # Exports the predictions to a polars Dataframe
    AP_df = export_predictions_to_df(imgs_path, batch_predictions, with_bboxes, confidence_columns, label_column_name)
    # Handles the corner case where AP_df is empty (because in this case it won't have any columns)
    if len(AP_df) == 0:
        AP_df = pl.DataFrame(schema=POLARS_AP_DF_SCHEMA)
        AP_df.drop_in_place("id")

    # Adds iou and match columns
    AP_df = AP_df.with_columns(iou=pl.lit(0.0, dtype=pl.Float32), match=pl.lit(0), correct_match=pl.lit(0))
    # Note: using with `with_row_count` instead of `with_row_index` because does not exist on Colab
    AP_df = AP_df.with_row_count("id")
    AP_df = AP_df.cast(POLARS_AP_DF_SCHEMA)

    # Matches images 1 by 1
    for img_path, img_detections, img_gts in zip(imgs_path, batch_predictions, batch_gts):

        # Handles the corner case where there are no predictions on the img
        if len(img_detections["probable_labels"]) == 0:
            unmatched_gts_labels = img_gts["labels"].detach().cpu().numpy().flatten()

        else:
            # Computes matching
            predictions_idx, gts_idx, ious = match_predictions_and_gts_for_one_img(
                img_detections["boxes"],
                img_gts["boxes"],
            )

            # Gets image predictions id
            img_predictions_id = AP_df.filter(pl.col("img_name") == img_path.name)["id"].to_numpy()
            # Gets predictions and gts labels
            img_detections_labels = img_detections["probable_labels"].detach().cpu().numpy()
            img_gts_labels = img_gts["labels"].detach().cpu().numpy().flatten()

            # For the matched predictions and gts saves the iou value and the match attribute (1=match, 0=no_match)
            AP_df = AP_df.with_columns(iou=AP_df["iou"].scatter(img_predictions_id[predictions_idx], ious))
            AP_df = AP_df.with_columns(match=AP_df["match"].scatter(img_predictions_id[predictions_idx], 1))

            # For the matched predictions and gts saves whether the match is a correct class match or not
            # (1=correct, 0=incorrect)
            mask_correct_matches = img_gts_labels[gts_idx] == img_detections_labels[predictions_idx]
            AP_df = AP_df.with_columns(
                correct_match=AP_df["correct_match"].scatter(
                    img_predictions_id[predictions_idx], mask_correct_matches.astype(int)
                )
            )

            # Gets the unmatched gts labels
            mask_unmatched_gts = np.ones(img_gts_labels.shape, dtype=bool)
            mask_unmatched_gts[gts_idx] = False
            unmatched_gts_labels = img_gts_labels[mask_unmatched_gts]

        # If there are some unmatched gts add them to the AP_df with null confidence, iou, match and correct_match.
        if len(unmatched_gts_labels) != 0:
            unmatched_gts_df = pl.DataFrame({"label": unmatched_gts_labels})
            unmatched_gts_df = unmatched_gts_df.with_columns(
                # Note: using 'pl.Utf8' instead of 'pl.String' because does not exist on Colab
                img_name=pl.lit(img_path.name, dtype=pl.Utf8),
                confidence=pl.lit(0.0, dtype=pl.Float32),
                iou=pl.lit(0.0, dtype=pl.Float32),
                match=pl.lit(0),
                correct_match=pl.lit(0),
            )
            # Note: using with `with_row_count` instead of `with_row_index` because does not exist on Colab
            unmatched_gts_df = unmatched_gts_df.with_row_count("id", offset=len(AP_df))

            unmatched_gts_df = unmatched_gts_df.select(AP_df.columns)
            unmatched_gts_df = unmatched_gts_df.cast(POLARS_AP_DF_SCHEMA)
            AP_df = pl.concat([AP_df, unmatched_gts_df], how="vertical")

    return AP_df


def training_matching_analysis(
    predictions: list[dict[str, np.ndarray | Tensor]],
    gts: list[dict[str, np.ndarray | Tensor]],
    matching_iou_threshold: float = 0.4,
):
    # Metrics initialization
    true_positives, false_positives, false_negatives = 0, 0, 0

    # Matches images 1 by 1
    for img_detections, img_gts in zip(predictions, gts):
        # Computes matching
        predictions_idx, gts_idx, ious = match_predictions_and_gts_for_one_img(
            img_detections["boxes"],
            img_gts["boxes"],
        )
        # Filters low iou matches
        predictions_idx, _, _ = filter_ids_with_low_iou(predictions_idx, gts_idx, ious, matching_iou_threshold)

        # Computes detection TP, FP and FN
        img_tp = len(predictions_idx)
        true_positives += img_tp
        n_predictions, n_gts = img_detections["probable_labels"].shape[0], img_gts["labels"].shape[0]
        false_positives += n_predictions - img_tp
        false_negatives += n_gts - img_tp

    return true_positives, false_positives, false_negatives
