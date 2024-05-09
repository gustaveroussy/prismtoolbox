import logging
import os
import uuid
import numpy as np
from shapely import MultiPolygon, Polygon
from shapely.geometry import mapping, shape
from shapely.affinity import translate
from shapely.ops import unary_union
from typing import Optional, Tuple, List, Union
from .data_utils import load_obj_with_json, save_obj_with_json


def contoursToPolygons(
    contours: List[np.ndarray], merge: Optional[bool] = False
) -> Union[Polygon, MultiPolygon]:
    """Converts list of arrays to shapely polygons.

    :param contours: list of contours to convert to shapely polygons
    :param offset: optional offset to add to each value in the arrays
    :param merge: optional boolean to merge the polygons
    :return: MultiPolygon object containing the polygons created from the arrays
    """
    polygons = [Polygon(contour.squeeze()).buffer(0) for contour in contours]
    result = []
    for poly in polygons:
        if poly.is_empty:
            continue
        if poly.geom_type == "MultiPolygon":
            result.append(max(poly.geoms, key=lambda x: x.area))
        else:
            result.append(poly)
    polygons = MultiPolygon(result)
    if merge:
        polygons = unary_union(polygons)
    return polygons


def PolygonsToContours(polygons: MultiPolygon):
    """Converts shapely polygons to list of arrays.

    :param polygons: shapely polygons to convert to arrays
    :return: list of contours containing the opencv-like contours created from the shapely polygons
    """
    return [
        np.array(poly.exterior.coords)[:-1, None, ...].astype(int)
        for poly in polygons.geoms
    ]


def read_qupath_annotations(path: str, offset: Optional[Tuple[int, int]] = (0, 0)):
    """Reads QuPath annotations from a .geojson file.

    :param path: path to the .geojson file
    :param offset: optional offset to add to each coordinate in the arrays
    :return: list of contours
    """
    data = load_obj_with_json(path)
    polygons = []
    for feature in data["features"]:
        if feature["geometry"]["type"] == "Polygon":
            polygons.append(shape(feature["geometry"]))
        elif feature["geometry"]["type"] == "MultiPolygon":
            polygons.extend(shape(feature["geometry"]).geoms)
        else:
            raise ValueError(
                "Feature type not recognized in .geojson file, please provide a .geojson file with only "
                "Polygon or MultiPolygon features."
            )
    polygons = MultiPolygon(polygons)
    polygons = translate(polygons, xoff=offset[0], yoff=offset[1])
    if not polygons.is_valid:
        polygons = polygons.buffer(0)
    return polygons


def convert_rgb_to_java_int_signed(rgb: Tuple[int, int, int]) -> int:
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
    polygons: MultiPolygon,
    path: str,
    object_type: str,
    offset: Optional[Tuple[int, int]] = (0, 0),
    label: Optional[str] = None,
    color: Optional[Tuple[int, int, int]] = None,
    append_to_existing_file: Optional[bool] = False,
):
    """Exports polygons to a .json or .geojson file.

    :param polygons: shapely polygons to export
    :param path: path to the .geojson file
    :param object_type: type of the object (should be either "annotation" or "detection")
    :param offset: optional offset to add to each coordinate in the arrays
    :param label: optional label of the polygons
    :param color: optional color of the polygons
    :param append_to_existing_file: optional boolean to append the polygons to an existing file
    """
    if isinstance(polygons, Polygon):
        polygons = MultiPolygon([polygons])
    # features = {"type": "FeatureCollection", "features": []}
    features = []
    properties = {"objectType": object_type}
    if label is not None:
        properties["classification"] = {
            "name": label,
            "colorRGB": convert_rgb_to_java_int_signed(color),
        }
    polygons = translate(polygons, xoff=offset[0], yoff=offset[1])
    for poly in polygons.geoms:
        # features.append({"type": "Feature", "geometry": mapping(poly)})
        features.append(
            {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": mapping(poly),
                "properties": properties,
            }
        )
    if os.path.exists(path) and append_to_existing_file:
        previous_features = load_obj_with_json(path)
        if len(previous_features) == 0:
            logging.warning(
                "The .geojson file does not contain any features, creating new file."
            )
        else:
            previous_features.extend(features)
            # previous_features["features"].extend(features["features"])
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
    if intersection.geom_type == "MultiPolygon":
        return intersection
    elif intersection.geom_type == "GeometryCollection":
        intersection = MultiPolygon(
            [poly for poly in intersection.geoms if isinstance(poly, Polygon)]
        )
    elif intersection.geom_type == "Polygon":
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
    merge: Optional[bool] = False,
) -> Union[Polygon, MultiPolygon]:
    """Converts patches to shapely polygons.

    :param patches: patches to convert to shapely polygons
    :param patch_size: size of the patches
    :param offset: optional offset to add to each coordinate in the arrays
    :param merge: optional boolean to merge the polygons
    :return: MultiPolygon object containing the polygons created from the patches
    """
    polygons = []
    ref_patch_size = patch_size * patch_downsample
    for patch in patches:
        x, y = patch
        polygons.append(
            Polygon(
                [
                    (x, y),
                    (x + ref_patch_size, y),
                    (x + ref_patch_size, y + ref_patch_size),
                    (x, y + ref_patch_size),
                ]
            )
        )
    polygons = MultiPolygon(polygons)
    if merge:
        polygons = unary_union(polygons)
    return polygons
