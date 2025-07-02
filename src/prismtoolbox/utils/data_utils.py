from __future__ import annotations

import json
import pickle
from typing import Any, Tuple

import geopandas as gpd
import h5py
import numpy as np


def save_obj_with_pickle(obj: object, file_path: str) -> None:
    """Save an object to a file using pickle.

    Args:
        obj: A pickeable object.
        file_path: The path to the file.
    """
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def save_obj_with_json(obj: object, file_path: str) -> None:
    """Save an object to a file using json.

    Args:
        obj: A json object.
        file_path: The path to the file.
    """
    with open(file_path, "w") as f:
        json.dump(obj, f)


def load_obj_with_pickle(file_path: str) -> Any:
    """Load an object from a file using pickle.

    Args:
        file_path: The path to the pickle file.

    Returns:
        A pickeable object from the file.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_obj_with_json(file_path: str) -> Any:
    """Load an object from a file using json.

    Args:
        file_path: The path to the json file.

    Returns:
        A json object from the file.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def read_h5_file(file_path: str, key: str) -> Tuple[np.ndarray, dict]:
    """Read an object from a h5 file.

    Args:
        file_path: The path to the h5 file.
        key: The key to select the dataset in the h5 file.

    Returns:
        A dataset from the h5 file.
    """
    with h5py.File(file_path, "r") as f:
        if key not in f:
            raise KeyError(f"Key '{key}' not found in the h5 file.")
        dataset = f[key]
        if isinstance(dataset, h5py.Dataset):
            obj = dataset[()]
            attrs = {k: v for k, v in dataset.attrs.items()}
        else:
            raise TypeError(f"Key '{key}' does not refer to a dataset in the h5 file.")
    return obj, attrs


def read_json_with_geopandas(
    file_path: str, offset: tuple[int, int] = (0, 0)
) -> gpd.GeoDataFrame:
    """Read a json file with geopandas.

    Args:
        file_path: The path to a json file.

    Returns:
        A GeoDataFrame object from the json file.
    """
    data = load_obj_with_json(file_path)
    df = gpd.GeoDataFrame.from_features(data)
    df.translate(xoff=offset[0], yoff=offset[1])
    if not df.is_valid.any():
        df.loc[~df.is_valid, :] = df.loc[~df.is_valid, :].buffer(0)
    if "classification" in df.columns:
        df["classification"] = df["classification"].apply(
            lambda x: x["name"] if type(x) == dict else x
        )
    return df
