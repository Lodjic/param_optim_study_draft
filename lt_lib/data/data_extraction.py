# Author : Loïc T

import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
from loguru import logger
from osgeo import gdal
from shapely.geometry import Polygon

###################################################################################
###################### Real annotations extraction functions ######################
###################################################################################


def get_geotransform_from_tiled_image_xml(xml_path: Path) -> list[float]:
    tree = ET.parse(xml_path)

    crs = tree.find("SRS").text.split('["', maxsplit=1)[1].split('"', maxsplit=1)[0]
    if crs != "WGS 84":
        logger.warning(f"CRS is different from WGS 84 for XML={xml_path.name}. Be carefull it uses CRS={crs}.")

    geotransform_coeffs = [float(x) for x in tree.find("GeoTransform").text.split(",")]

    return crs, geotransform_coeffs


def convert_geobboxes_to_pixels(geobboxes_list: list[list[float]], geotransform_coeffs) -> list[list[float]]:
    bboxes = []
    for bbox in geobboxes_list:
        pixel_bbox = list(map(int, gdal.ApplyGeoTransform(gdal.InvGeoTransform(geotransform_coeffs), bbox[0], bbox[1])))
        pixel_bbox += list(
            map(int, gdal.ApplyGeoTransform(gdal.InvGeoTransform(geotransform_coeffs), bbox[2], bbox[3]))
        )
        # Re-arrange the pixels coords to have the top-left first and bottom-right then (pascal_voc format)
        pixel_bbox = [
            min(pixel_bbox[0], pixel_bbox[2]),
            min(pixel_bbox[1], pixel_bbox[3]),
            max(pixel_bbox[0], pixel_bbox[2]),
            max(pixel_bbox[1], pixel_bbox[3]),
        ]
        pixel_bbox = list(map(lambda x: min(max(x, 0), 511), pixel_bbox))
        bboxes.append(pixel_bbox)
    return bboxes


def get_annotations_dataframe_from_geojson(
    image_name: str,
    geojson_path: Path,
    xml_path: Path,
) -> list[list[float]]:
    gdf = gpd.read_file(geojson_path)
    polygons_list = gdf["geometry"].to_list()
    geobboxes_list = [polygon.bounds for polygon in polygons_list]
    crs, geotransform = get_geotransform_from_tiled_image_xml(xml_path)
    bboxes_array = np.array(convert_geobboxes_to_pixels(geobboxes_list, geotransform))

    if not np.all((bboxes_array >= 0) & (bboxes_array < 512)):
        logger.warning(
            f"A bbox from geojson '{geojson_path.parent.name}/{geojson_path.name}' is out of bounds with values: {bboxes_array}"
        )

    gdf["label"] = gdf["role_id"]
    gdf["sub_id"] = np.arange(bboxes_array.shape[0])
    gdf["bbox_xmin"] = bboxes_array[:, 0]
    gdf["bbox_ymin"] = bboxes_array[:, 1]
    gdf["bbox_xmax"] = bboxes_array[:, 2]
    gdf["bbox_ymax"] = bboxes_array[:, 3]
    gdf["img_name"] = [image_name] * bboxes_array.shape[0]
    gdf["img_crs"] = [crs] * bboxes_array.shape[0]
    gdf["geotransform0"] = [geotransform[0]] * bboxes_array.shape[0]
    gdf["geotransform1"] = [geotransform[1]] * bboxes_array.shape[0]
    gdf["geotransform2"] = [geotransform[2]] * bboxes_array.shape[0]
    gdf["geotransform3"] = [geotransform[3]] * bboxes_array.shape[0]
    gdf["geotransform4"] = [geotransform[4]] * bboxes_array.shape[0]
    gdf["geotransform5"] = [geotransform[5]] * bboxes_array.shape[0]

    return gdf


##### Unused functions #####


def get_polygon_coords_from_geojson(geojson_path: Path) -> list[list[float]]:
    polygons_list = get_polygons_list_from_geojson(geojson_path)

    polygons_list = []
    for polygon in polygons_list:
        coords = list(polygon.exterior.coords)
        if len(coords) != 5:
            logger.warning(f"There were {len(coords)} points in a Polygon in {geojson_path}. There should be 5!")
        polygons_list.append(coords)

    return polygons_list


def get_polygons_list_from_geojson(geojson_path: Path) -> list[Polygon]:
    gdf = gpd.read_file(geojson_path)
    return gdf["geometry"].to_list()


def get_geobboxes_from_geojson(geojson_path: Path) -> list[list[float]]:
    polygons_list = get_polygons_list_from_geojson(geojson_path)
    bboxes = [polygon.bounds for polygon in polygons_list]
    return bboxes


def get_pixel_bboxes_from_geojson(geojson_path: Path, xml_path: Path) -> list[list[float]]:
    geobboxes_list = get_geobboxes_from_geojson(geojson_path)
    _, geotransform = get_geotransform_from_tiled_image_xml(xml_path)
    bboxes_list = convert_geobboxes_to_pixels(geobboxes_list, geotransform)

    if not np.all((np.array(bboxes_list) >= 0) & (np.array(bboxes_list) < 512)):
        logger.warning(f"A bbox from geojson '{geojson_path}' is out of bounds with values: {bboxes_list}")

    return bboxes_list


########################################################################################
###################### Synthetic annotations extraction functions ######################
########################################################################################


def get_bbox_and_wingspan_from_synthetic_xml_object(
    xml_object: ET.ElementTree, resolution: list[int] | tuple[int], clip: bool = True
) -> tuple[list[float], float]:
    diamond_points = []
    diamond_points.append(xml_object.find("Sockets").find("Bone_PlaneAnnotation_Nose").find("screen").text)
    diamond_points.append(xml_object.find("Sockets").find("Bone_PlaneAnnotation_Tail").find("screen").text)
    diamond_points.append(xml_object.find("Sockets").find("Bone_PlaneAnnotation_RightWing").find("screen").text)
    diamond_points.append(xml_object.find("Sockets").find("Bone_PlaneAnnotation_LeftWing").find("screen").text)

    for i, point in enumerate(diamond_points):
        diamond_points[i] = [float(coord.split("=")[1]) for coord in point.split(" ")]
    diamond_points = np.array(diamond_points)

    wingspan = np.linalg.norm(diamond_points[2, :] - diamond_points[3, :])

    bbox = [
        int(xml_object.find("bndbox2D").find("xmin").text),
        int(xml_object.find("bndbox2D").find("ymin").text),
        int(xml_object.find("bndbox2D").find("xmax").text),
        int(xml_object.find("bndbox2D").find("ymax").text),
    ]

    if not clip:
        extrapolated_bbox = [
            np.floor(min(diamond_points[:, 0])),
            np.floor(min(diamond_points[:, 1])),
            np.ceil(max(diamond_points[:, 0])),
            np.ceil(max(diamond_points[:, 1])),
        ]

        for i, extrapolated_coord in enumerate(extrapolated_bbox[:2]):
            if extrapolated_bbox < 0:
                bbox[i] = extrapolated_coord
        if extrapolated_bbox[2] > resolution[0]:
            bbox[2] = extrapolated_bbox[2]
        if extrapolated_bbox[3] > resolution[1]:
            bbox[3] = extrapolated_bbox[3]

    return bbox, wingspan


def get_bboxes_df_from_one_synthetic_xml(
    xml_path: Path,
    img_name: str,
    clip: bool = True,
    wingspan_label_bins: list[float] = [0, 15, 36],
    imgs_resolution: float = 0.3,
) -> pl.DataFrame:
    tree = ET.parse(xml_path)
    resolution = (
        int(tree.find("image_resolution").find("width").text),
        int(tree.find("image_resolution").find("height").text),
    )
    objects = tree.findall("object")

    # Default bins values was obtained from the wingspan thresholds given by the rareplanes public user guide. Bin
    # values are divided by 0.3 because synthetic pixels have a resolution of approximately 0.3m
    wingspan_label_bins = np.array(wingspan_label_bins) / imgs_resolution
    bboxes = []

    for object in objects:
        if object.find("category0").text == "Airplane":
            if object.find("category1").text == "Civil":
                bbox, wingspan = get_bbox_and_wingspan_from_synthetic_xml_object(object, resolution, clip)
                bboxes.append([img_name, *bbox, np.digitize(wingspan, wingspan_label_bins), wingspan])
            else:
                logger.warning(
                    "The 'category1' attribute is not Civil but "
                    + f"{object.find('category1').text}-{object.find('category2').text}"
                )

    bboxes_df = pl.from_records(
        bboxes, schema=["img_name", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "label", "wingspan"]
    )
    bboxes_df = bboxes_df.with_row_index("sub_id")

    return bboxes_df


###########################################################################################
###################### Annotations extraction functions from gts.csv ######################
###########################################################################################


def get_img_bboxes_from_csv(csv_path: Path, img_name: str):
    gts_df = pl.read_csv(csv_path)
    gts_df = gts_df.filter(pl.col("img_name") == img_name)
    bboxes = gts_df[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].to_numpy()
    labels = gts_df["label"].to_list()
    return bboxes, labels
