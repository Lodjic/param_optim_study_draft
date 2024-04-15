# Author : Loïc Thiriet

import os
import shutil
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger
from tqdm.auto import tqdm

from lt_lib.data.data_extraction import (
    get_annotations_dataframe_from_geojson,
    get_bboxes_df_from_one_synthetic_xml,
)
from lt_lib.data.data_utils import (
    construct_similar_file_path_list_from_another_dir,
    copy_file_list,
    move_file_list,
    split_files_randomly_from_list,
)
from lt_lib.data.datasets import POLARS_GTS_SCHEMA
from lt_lib.utils.load_and_save import save_dict_as_json

#################################################################################
###################### Functions to extract and gather gts ######################
#################################################################################


### Functions for real data


def get_all_annotations_from_rareplanes_geojsons(
    root_dir_path: Path, tiled_version: bool = True, imgs_extension: str = ".png", save_to_file: bool = True
) -> tuple[gpd.GeoDataFrame, dict[str : dict[str : list[int]]]]:
    logger.info("Launching the gathering of all annotations into 1 unique geojson file.")
    # Defining paths
    images_dir_path = root_dir_path / "imgs"
    geojsons_dir_path = root_dir_path / "annotations"

    # Construct the geodf containing all the gts
    global_gdf = gpd.GeoDataFrame()
    if tiled_version:
        for image_path in tqdm(list(images_dir_path.glob(f"*{imgs_extension}"))):
            # Construct necessary paths and file names
            xml_path = image_path.with_suffix(f"{imgs_extension}.aux.xml")
            geojson_path = geojsons_dir_path / image_path.with_suffix(".geojson").name
            # Get the annotations dataframe
            gdf = get_annotations_dataframe_from_geojson(image_path.name, geojson_path, xml_path)
            global_gdf = pd.concat([global_gdf, gdf], ignore_index=True)
    else:
        raise ValueError("Annotations extraction from the non tiled dataset folder is not yet implemented.")
    global_gdf["id"] = np.arange(len(global_gdf))

    # Exctract a minimal dataframe containing only the labels and bboxes positions
    minimal_gts_df = pl.from_pandas(
        global_gdf[["id", "sub_id", "img_name", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "label"]],
        schema_overrides=POLARS_GTS_SCHEMA,
    )

    # Constructing a json version of the minimal dataframe
    minimal_gts_json = {}
    for img_name in np.unique(minimal_gts_df["img_name"].to_list()):
        sub_img_df = minimal_gts_df.filter(pl.col("img_name") == img_name)
        minimal_gts_json[img_name] = {
            "ids": sub_img_df["id"].to_list(),
            "bboxes": sub_img_df[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].to_numpy().tolist(),
            "labels": sub_img_df["label"].to_list(),
        }

    # Save the files into 2 csv and 1 json + 1 json dict
    if save_to_file:
        global_gdf.to_file(geojsons_dir_path / "global_annotations.geojson", driver="GeoJSON")
        # minimal_gts_df.to_csv(geojsons_dir_path / "gts.csv", index=False)
        minimal_gts_df.write_csv(geojsons_dir_path / "gts.csv", include_header=True)
        save_dict_as_json(geojsons_dir_path / "gts.json", minimal_gts_json)

    return global_gdf, minimal_gts_json


### Functions for synthetic data


def get_all_synthetic_gts_from_rareplanes_xmls(
    annotations_dir_path: Path,
    imgs_extension: str = ".png",
    clip: bool = True,
    wingspan_label_bins: list[float] = [0, 15, 36],
    imgs_resolution: float = 0.3,
    save_to_file: bool = True,
) -> pl.DataFrame:
    bboxes_df = pl.DataFrame()
    for xml_path in tqdm(list(annotations_dir_path.glob("*.xml"))):
        bboxes_df = pl.concat(
            [
                bboxes_df,
                get_bboxes_df_from_one_synthetic_xml(
                    xml_path, xml_path.with_suffix(imgs_extension).name, clip, wingspan_label_bins, imgs_resolution
                ),
            ]
        )

    bboxes_df = bboxes_df.with_row_index("id")

    # Save the file into a csv
    if save_to_file:
        bboxes_df.write_csv(annotations_dir_path / "gts.csv", include_header=True)

    return bboxes_df


##############################################################################
###################### Functions to run train-val split ######################
##############################################################################


### Functions for real data


def get_rareplanes_real_xml_and_geojson_lists(img_list: list) -> tuple[list[Path], list[Path]]:
    xml_list = construct_similar_file_path_list_from_another_dir(
        img_list, img_list[0].parent, f"{img_list[0].suffix}.aux.xml"
    )
    geojson_list = construct_similar_file_path_list_from_another_dir(
        img_list, img_list[0].parent.parent / "annotations", f".geojson"
    )
    return xml_list, geojson_list


def move_rareplanes_real_img_xml_and_geojson_files(
    img_list: list[Path], xml_list: list[Path], geojson_list: list[Path], destination_dir_name: Literal["train", "val"]
) -> None:
    root_imgs_dir = img_list[0].parent.parent

    # Creates train and val directories
    if not os.path.isdir(root_imgs_dir / f"{destination_dir_name}/imgs"):
        os.makedirs(root_imgs_dir / f"{destination_dir_name}/imgs")
    if not os.path.isdir(root_imgs_dir / f"{destination_dir_name}/annotations"):
        os.mkdir(root_imgs_dir / f"{destination_dir_name}/annotations")

    # Moves files
    move_file_list(img_list, root_imgs_dir / f"{destination_dir_name}/imgs")
    move_file_list(xml_list, root_imgs_dir / f"{destination_dir_name}/imgs")
    move_file_list(geojson_list, root_imgs_dir / f"{destination_dir_name}/annotations")


def train_val_split_and_move_rareplanes_real_img_xml_and_geojson_files(
    imgs_dir: Path, imgs_extension: str, val_proportion: float, seed: int | None = 42
) -> tuple[list[Path], list[Path]]:
    # Splits images and annotations files
    img_list = list(imgs_dir.glob(f"*{imgs_extension}"))
    train_img_list, val_img_list = split_files_randomly_from_list(img_list, val_proportion, seed)
    train_xml_list, train_geojson_list = get_rareplanes_real_xml_and_geojson_lists(train_img_list)
    val_xml_list, val_geojson_list = get_rareplanes_real_xml_and_geojson_lists(val_img_list)

    # Moves files in the correct directories
    move_rareplanes_real_img_xml_and_geojson_files(train_img_list, train_xml_list, train_geojson_list, "train")
    move_rareplanes_real_img_xml_and_geojson_files(val_img_list, val_xml_list, val_geojson_list, "val")


### Functions for synthetic data


def copy_rareplanes_synthetic_img_and_xml_files(
    img_list: list[Path], xml_list: list[Path], destination_root_data_dir: Path | None = None
) -> None:
    root_data_dir = img_list[0].parent.parent

    if not destination_root_data_dir:
        destination_root_data_dir = root_data_dir / "sampled_synthetic_data"

    # Creates imgs and annotations directories
    if not os.path.isdir(destination_root_data_dir / "imgs"):
        os.makedirs(destination_root_data_dir / "imgs")
    if not os.path.isdir(destination_root_data_dir / "annotations"):
        os.mkdir(destination_root_data_dir / "annotations")

    # Moves files
    copy_file_list(img_list, destination_root_data_dir / "imgs")
    copy_file_list(xml_list, destination_root_data_dir / "annotations")


def copy_fraction_of_random_rareplanes_synthetic_img_and_xml_files(
    imgs_dir: Path,
    imgs_extension: str,
    fraction_to_copy: float,
    seed: int | None = 42,
    destination_dir: Path | None = None,
) -> tuple[list[Path], list[Path]]:
    # Splits images and annotations files
    img_list = list(imgs_dir.glob(f"*{imgs_extension}"))
    _, img_list_to_copy = split_files_randomly_from_list(img_list, fraction_to_copy, seed)
    xml_list_to_copy = construct_similar_file_path_list_from_another_dir(
        img_list_to_copy, img_list_to_copy[0].parent.parent.parent / "annotations", ".xml"
    )

    # Copies files in the correct directories
    copy_rareplanes_synthetic_img_and_xml_files(img_list_to_copy, xml_list_to_copy, destination_dir)


def copy_nb_of_random_rareplanes_synthetic_tile_and_gts_files(
    root_data_dir: Path,
    imgs_extension: str,
    select_nb: int,
    destination_root_data_dir: Path | None = None,
    seed: int | None = 42,
) -> tuple[list[Path], list[Path]]:
    # Get the image names that have objects on them
    gts_df = pl.read_csv(root_data_dir / "annotations/gts.csv")
    gts_df = gts_df.drop_nulls()
    img_name_list = gts_df["img_name"].unique().to_list()
    # Randomly selects image names
    _, selected_img_name_list = split_files_randomly_from_list(img_name_list, select_nb, seed)
    # Filters the gts_df to keep only the gts of the selected images
    gts_df = gts_df.filter(pl.col("img_name").is_in(selected_img_name_list))

    # Constructs the images' paths
    selected_img_path_list = construct_similar_file_path_list_from_another_dir(
        file_list=list(map(Path, selected_img_name_list)), directory=root_data_dir / "imgs", extension=imgs_extension
    )

    # Infer a destinations dir if not specified
    if not destination_root_data_dir:
        destination_root_data_dir = root_data_dir / "sampled_synthetic_tiled_data"

    # Creates imgs and annotations directories
    if not os.path.isdir(destination_root_data_dir / "imgs"):
        os.makedirs(destination_root_data_dir / "imgs")
    if not os.path.isdir(destination_root_data_dir / "annotations"):
        os.mkdir(destination_root_data_dir / "annotations")

    # Copies imgs in the correct directory
    copy_file_list(selected_img_path_list, destination_root_data_dir / "imgs")
    # Writes the gts_df the annotations directory
    gts_df.write_csv(destination_root_data_dir / "annotations/gts.csv")


##################################################################################
###################### Functions to adjust synthetic labels ######################
##################################################################################


def adjust_labels_base_on_wingspan(
    gts_with_wingspan_path: Path,
    gts_to_adjust_path: Path,
    wingspan_label_bins: list[float] = [0, 15, 36],
    imgs_resolution: float = 0.3,
    save_to_file: bool = True,
) -> None:
    gts_with_wingspan = pl.read_csv(gts_with_wingspan_path)
    gts_to_adjust = pl.read_csv(gts_to_adjust_path)

    # Bins values can be chosen with the help of the wingspan thresholds given by the rareplanes public user guide and
    # experiments. Bin values are divided by 0.3 because synthetic pixels have a resolution of 0.3m
    wingspan_label_bins = np.array(wingspan_label_bins) / imgs_resolution

    for row in tqdm(gts_to_adjust.iter_rows(named=True), total=len(gts_to_adjust)):
        id = row["id"]
        row_with_wingspan = gts_with_wingspan.row(by_predicate=pl.col("id") == id, named=True)
        row["label"] = np.digitize(row_with_wingspan["wingspan"], wingspan_label_bins)

    if save_to_file:
        gts_to_adjust.write_csv(gts_to_adjust_path.parent / "gts_adjusted.csv")
