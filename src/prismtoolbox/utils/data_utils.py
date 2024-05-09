import pickle
import json
import h5py
import numpy as np
from typing import Tuple, Any


def save_obj_with_pickle(obj: object, file_path: str) -> None:
    """
    Save an object to a file using pickle.
    :param obj: a pickeable object
    :param file_path: path to the file
    """
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def save_obj_with_json(obj: object, file_path: str) -> None:
    """
    Save an object to a file using json.
    :param obj: a json object
    :param file_path: path to the file
    """
    with open(file_path, "w") as f:
        json.dump(obj, f)


def load_obj_with_pickle(file_path: str) -> Any:
    """
    Load an object from a file using pickle.
    :param file_path: path to a pickle file
    :return: a pickeable object from the file
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_obj_with_json(file_path: str) -> Any:
    """
    Load an object from a file using json.
    :param file_path: path to a json file
    :return: a json object from the file
    """
    with open(file_path, "r") as f:
        return json.load(f)


def read_h5_file(file_path: str, key: str) -> Tuple[np.ndarray, dict]:
    """
    Read an object from a h5 file.
    :param file_path: path to a h5 file
    :param key: key to the object in the h5 file
    :return: an object from the h5 file
    """
    with h5py.File(file_path, "r") as f:
        object = f[key][()]
        attrs = {key: value for key, value in f[key].attrs.items()}
    return object, attrs
