from __future__ import annotations

import logging
import os
import uuid

import numpy as np
from shapely import MultiPolygon, Polygon, box
from shapely.affinity import translate
from shapely.geometry import mapping
from shapely.ops import unary_union

from .data_utils import load_obj_with_json, read_json_with_geopandas, save_obj_with_json

log = logging.getLogger(__name__)


def contoursToPolygons(
    contours: list[np.ndarray],
    merge: bool = False,
    make_valid: bool = False,
) -> MultiPolygon:
    """Converts list of arrays to shapely polygons.

    :param contours: list of contours to convert to shapely polygons
    :param merge: optional boolean to merge the polygons
    :param make_valid: optional boolean to enforce validity of the polygons
    :return: MultiPolygon object containing the polygons created from the arrays
    """
    polygons = [Polygon(contour.squeeze()).buffer(0) for contour in contours]
    result = []
    for poly in polygons:
        if poly.is_empty:
            continue
        if isinstance(poly, MultiPolygon):
            result.append(max(poly.geoms, key=lambda x: x.area))
        else:
            result.append(poly)
    polygons = MultiPolygon(result)
    if make_valid and not polygons.is_valid:
        buffered = polygons.buffer(0)
        if isinstance(buffered, Polygon):
            polygons = MultiPolygon([buffered])
    if merge:
        polygons = unary_union(polygons)
        if isinstance(polygons, Polygon):
            polygons = MultiPolygon([polygons])
    if not isinstance(polygons, MultiPolygon):
        raise ValueError("Resulting polygons are not a MultiPolygon.")
    return polygons


def PolygonsToContours(polygons: MultiPolygon) -> list[np.ndarray]:
    """Converts shapely polygons to list of arrays.

    :param polygons: shapely polygons to convert to arrays
    :return: list of contours containing the opencv-like contours created from the shapely polygons
    """
    return [
        np.array(poly.exterior.coords)[:-1, None, ...].astype(int)
        for poly in polygons.geoms
    ]


def read_qupath_annotations(
    path: str,
    offset: tuple[int, int] = (0, 0),
    class_name: str = "annotation",
    column_to_select: str = "objectType",
) -> MultiPolygon:
    """Reads pathologist annotations from a .geojson file.

    :param path: path to the .geojson file
    :param offset: optional offset to add to each coordinate in the arrays
    :param class_name: name of the class to select
    :param column_to_select: optional column to select
    :return: MultiPolygon object containing the polygons of the selected class.
    """
    df = read_json_with_geopandas(path, offset)
    polygons = []
    for poly in df.loc[df[column_to_select] == class_name, "geometry"].values:
        if poly.geom_type == "Polygon":
            polygons.append(poly)
        elif poly.geom_type == "MultiPolygon":
            polygons.extend(poly.geoms)
        else:
            raise ValueError("Geometry type not supported.")
    polygons = MultiPolygon(polygons)
    if not polygons.is_valid:
        buffered = polygons.buffer(0)
        if isinstance(buffered, Polygon):
            polygons = MultiPolygon([buffered])
        if not isinstance(polygons, MultiPolygon):
            raise ValueError("Resulting polygons are not a MultiPolygon.")
    return polygons


def convert_rgb_to_java_int_signed(rgb: tuple[int, int, int]) -> int:
    """Converts RGB tuple to Java signed integer.

    :param rgb: RGB tuple
    :return: Java signed integer
    """
    r, g, b = rgb
    java_rgb = (255 << 24) | (r << 16) | (g << 8) | b
    if java_rgb >= (1 << 31):
        java_rgb -= 1 << 32
    return java_rgb


def export_polygons_to_qupath(
    polygons: MultiPolygon | Polygon,
    path: str,
    object_type: str,
    offset: tuple[int, int] = (0, 0),
    label: str | None = None,
    color: tuple[int, int, int] | None = None,
    append_to_existing_file: bool = False,
    as_feature_collection: bool = False,
):
    """Exports polygons to a .json or .geojson file.

    :param polygons: shapely polygons to export
    :param path: path to the .geojson file
    :param object_type: type of the object (should be either "annotation" or "detection")
    :param offset: optional offset to add to each coordinate in the arrays
    :param label: optional label of the polygons
    :param color: optional color of the polygons
    :param append_to_existing_file: optional boolean to append the polygons to an existing file
    :param as_feature_collection: optional boolean to save the polygons as a FeatureCollection
    """
    if isinstance(polygons, Polygon):
        polygons = MultiPolygon([polygons])
    features = []
    properties = {}
    properties["objectType"] = object_type
    if label is not None:
        if color is None:
            log.warning(
                "No color provided for the label, using default color (255, 0, 0)."
            )
            color = (255, 0, 0)
        properties["classification"] = {
            "name": label,
            "colorRGB": convert_rgb_to_java_int_signed(color),
        }
    polygons = translate(polygons, xoff=offset[0], yoff=offset[1])
    for poly in polygons.geoms:
        features.append(
            {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": mapping(poly),
                "properties": properties,
            }
        )
    features = (
        {"type": "FeatureCollection", "features": features}
        if as_feature_collection
        else features
    )
    if os.path.exists(path) and append_to_existing_file:
        previous_features = load_obj_with_json(path)
        if len(previous_features) == 0:
            log.warning(
                "The .geojson file does not contain any features, creating new file."
            )
        else:
            if as_feature_collection:
                previous_features["features"].extend(features["features"]) # type: ignore
            else:
                previous_features.extend(features)
            features = previous_features
    save_obj_with_json(features, path)


def intersectionPolygons(
    polygons1: MultiPolygon, polygons2: MultiPolygon
) -> MultiPolygon:
    """Computes the intersection of two MultiPolygons.

    :param polygons1: first MultiPolygon
    :param polygons2: second MultiPolygon
    :return: MultiPolygon containing the intersection of the two input MultiPolygons
    """
    intersection = polygons1 & polygons2
    if isinstance(intersection, MultiPolygon):
        return intersection
    elif intersection.geom_type == "GeometryCollection":
        intersection = MultiPolygon(
            [poly for poly in intersection.geoms if isinstance(poly, Polygon)] # type: ignore
        )
    elif isinstance(intersection, Polygon):
        intersection = MultiPolygon([intersection])
    else:
        raise ValueError(
            "Intersection of provided MultiPolygons is not a MultiPolygon or a Polygon"
        )
    return intersection


def patchesToPolygons(
    patches: np.ndarray,
    patch_size: int,
    patch_downsample: int,
    merge: bool = False,
) -> MultiPolygon:
    """Converts patches to shapely polygons.

    :param patches: Top left point coordinates of the patches to convert to shapely polygons
    :param patch_size: size of the patches
    :param merge: optional boolean to merge the polygons
    :return: MultiPolygon object containing the polygons created from the patches
    """
    polygons = []
    ref_patch_size = patch_size * patch_downsample
    for patch in patches:
        x, y = patch
        polygons.append(box(x, y, x + ref_patch_size, y + ref_patch_size, ccw=False))
    polygons = MultiPolygon(polygons)
    if merge:
        polygons = unary_union(polygons)
        if isinstance(polygons, Polygon):
            polygons = MultiPolygon([polygons])
    if not isinstance(polygons, MultiPolygon):
        raise ValueError("Resulting polygons are not a MultiPolygon.")
    return polygons
