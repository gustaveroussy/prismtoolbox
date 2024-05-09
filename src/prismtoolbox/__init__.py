import importlib.metadata

# __version__ = importlib.metadata.version("prismtoolbox")

from .wsicore import WSI
from .utils import data_utils, qupath_utils, vis_utils

__all__ = ["WSI", "data_utils", "qupath_utils", "vis_utils"]
