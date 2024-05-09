from __future__ import annotations

import torch
import shapely
import numpy as np
from shapely import Polygon, MultiPolygon
from tqdm import tqdm
from functools import partial
from cellpose import models as cp_models
from .models import create_sop_segmenter, create_sop_postprocessing


def create_cellpose_tools(
    device: str = "cuda", 
    model_type: str = "cyto3", 
    min_size=15, 
    flow_threshold=0.4, 
    channel_cellpose=0
):
    model = cp_models.Cellpose(model_type=model_type, device=torch.device(device))
    model_infer = partial(
        model.eval,
        min_size=min_size,
        flow_threshold=flow_threshold,
        channels=[channel_cellpose, 0],
        diameter=30,
        invert=False,
    )
    preprocessing_fct = lambda x: x.permute(0, 2, 3, 1).numpy()
    postprocessing_fct = lambda x: x[0]
    return preprocessing_fct, model_infer, postprocessing_fct


def create_sop_tools(
    device,
    model_type="unet_256",
    norm="instance",
    pretrained_weights=None,
    sigma=1,
    disk_size=12,
    erosion=True,
):
    model = create_sop_segmenter(model_type, norm, pretrained_weights)
    model.to(device)
    model.eval()
    preprocessing_fct = lambda x: x.to(device)
    postprocessing_fct = create_sop_postprocessing(sigma, disk_size, erosion)
    return preprocessing_fct, model, postprocessing_fct


def create_segmentation_tools(
    model_name, pretrained_weights: str | None = None, device: str = "cuda", **kwargs
):
    if model_name == "cellpose":
        return create_cellpose_tools(device, **kwargs)
    elif model_name == "sop":
        return create_sop_tools(device, pretrained_weights=pretrained_weights, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def solve_conflicts(
    cells: list[Polygon],
    threshold: float = 0.5,
    return_indices: bool = False,
) -> np.ndarray[Polygon] | tuple[np.ndarray[Polygon], np.ndarray]:
    """Resolve segmentation conflicts (i.e. overlap) after running segmentation on patches

    Args:
        cells: List of cell polygons
        threshold: When two cells are overlapping, we look at the area of intersection over the area of the smallest
         cell. If this value is higher than the `threshold`, the cells are merged
        return_indices: If `True`, returns also the cells indices. Merged cells have an index of -1.

    Returns:
        Array of resolved cells polygons. If `return_indices`, it also returns an array of cell indices.
    """
    cells = list(cells)
    n_cells = len(cells)
    resolved_indices = np.arange(n_cells)

    assert n_cells > 0, "No cells was segmented, cannot continue"

    tree = shapely.STRtree(cells)
    conflicts = tree.query(cells, predicate="intersects")

    conflicts = conflicts[:, conflicts[0] != conflicts[1]].T
    conflicts = np.array([c for c in conflicts if c[0] < c[1]])

    for i1, i2 in tqdm(conflicts, desc="Resolving conflicts"):
        resolved_i1 = resolved_indices[i1]
        resolved_i2 = resolved_indices[i2]
        cell1, cell2 = cells[resolved_i1], cells[resolved_i2]

        intersection = cell1.intersection(cell2).area
        if intersection >= threshold * min(cell1.area, cell2.area):
            cell = cell1 | cell2

            resolved_indices[np.isin(resolved_indices, [resolved_i1, resolved_i2])] = len(
                cells
            )
            cells.append(cell)

    unique_indices = np.unique(resolved_indices)
    unique_cells = MultiPolygon(list(np.array(cells)[unique_indices]))

    if return_indices:
        return unique_cells, np.where(unique_indices < n_cells, unique_indices, -1)

    return unique_cells
