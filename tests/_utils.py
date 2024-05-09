import os
import h5py
import json
import pickle


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
