import os
import h5py
import json
import pickle
import torch
import numpy as np
from shapely import box
from shapely.geometry import mapping
from shapely.affinity import translate

from prismtoolbox.utils.data_utils import save_obj_with_json

def check_pickle_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            try:
                data = pickle.load(f)
            except Exception as e:
                return False
            if len(data) > 0 and len(data[0]) > 0:
                return True
    return False


def check_h5_file(file_path):
    if os.path.exists(file_path):
        try:
            data = h5py.File(file_path, "r")
        except Exception as e:
            return False
        if len(data["coords"][:]) > 0 and len(data["coords"].attrs.keys()) > 0:
            return True
    return False


def check_geojson_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                return False
            if len(data) > 0 and data[0]["id"] is not None and len(data[0]["geometry"]["coordinates"]) > 0:
                return True
    return False

def load_embeddings(file_path, format):
    if format == "pt":
        return torch.load(file_path)
    elif format == "npy":
        return np.load(file_path)
    else:
        raise ValueError(f"Unknown format {format}")

def generate_random_boxes(coord, patch_size, n_boxes):
    x, y = coord
    x_max = x + patch_size
    y_max = y + patch_size
    boxes = []
    for i in range(n_boxes):
        x1 = np.random.randint(x, x_max)
        y1 = np.random.randint(y, y_max)
        x2 = np.random.randint(x1+1, x_max+1)
        y2 = np.random.randint(y1+1, y_max+1)
        boxes.append(box(x1, y1, x2, y2, ccw=True))
    return boxes
    
def generate_fake_cell_segmentation(path, coords, patch_size, cell_classes, n_cells_by_patch, offset=None):
    features = []
    for k, cell_class in enumerate(cell_classes):
        for i, coord in enumerate(coords):
            cells = generate_random_boxes(coord, patch_size, n_cells_by_patch)
            for cell in cells:
                if offset is not None:
                    cell = translate(cell, xoff=offset[0], yoff=offset[1])
                features.append(
                {
                "type": "Feature",
                "id": str(k+i),
                "geometry": mapping(cell),
                "properties": {"classification": cell_class},
                }
                )
    return save_obj_with_json(features, path)
            