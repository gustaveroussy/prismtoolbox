from __future__ import annotations
import numpy as np
import geopandas as gpd
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transformsv2
import sklearn.cluster as skl_cluster
from functools import partial
from shapely import box
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from tqdm import tqdm

from .models import (
    create_torchvision_embedder,
    create_clam_embedder,
    create_pathoduet_embedder,
    create_transformers_embedder,
    create_timm_embedder,
    create_conch_embedder,
)

from prismtoolbox.utils.stain_utils import retrieve_conv_matrix

arch_dict = {
    "resnet18": partial(create_torchvision_embedder, name="ResNet18"),
    "resnet50": partial(create_torchvision_embedder, name="ResNet50"),
    "resnet101": partial(create_torchvision_embedder, name="ResNet101"),
    "clam": create_clam_embedder,
    "pathoduet": create_pathoduet_embedder,
    "phikon": partial(
        create_transformers_embedder,
        "owkin/phikon",
        pretrained=True,
        add_pooling_layer=False,
    ),
    "uni": partial(
        create_timm_embedder,
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    ),
    "conch": create_conch_embedder,
}

clustering_metrics = {
    "silhouette": silhouette_score,
    "calinski_harabasz": calinski_harabasz_score,
    "davies_bouldin": davies_bouldin_score,
}

CELL_FEATURE_NAMES = [
    "N_cells",
    "avg_cells_area",
    "avg_cells_perimeter",
    "avg_cells_compactness",
    "avg_cells_roundness",
    "avg_cells_solidity",
    "avg_cells_elongation",
]

STAIN_FEATURE_NAMES = [
    "avg",
    "std",
    "min",
    "max",
    "median",
    "q1",
    "q3",
]

def create_model(
    arch_name: str,
    pretrained_weights: str | None = None
) -> tuple[nn.Module, transformsv2.Compose | None]:
    """Create a model from the arch_name and load the pretrained weights if specified.

    Args:
        arch_name: The architecture name. At the moment, possible models are:

            - [torchvision based models](https://pytorch.org/vision/stable/models.html): "resnet18", "resnet50",
             "resnet101".
            - [CLAM](https://github.com/mahmoodlab/CLAM), i.e ResNet50 truncated after the third convolutional
            block: "clam".
            - [Pathoduet model](https://github.com/openmedlab/PathoDuet): "pathoduet".
            - [Phikon model](https://huggingface.co/owkin/phikon): "phikon".
            - [Conch model](https://huggingface.co/MahmoodLab/CONCH): "conch".
            - [Uni model](https://huggingface.co/MahmoodLab/UNI): "uni".

        pretrained_weights: The path to the pretrained weights or the name of the pretrained weights.

            - For torchvision models and CLAM, possible weights are: "IMAGENET1K_V1", "IMAGENET1K_V2", and
            "[ciga](https://github.com/ozanciga/self-supervised-histopathology)" for ResNet18.
            - For Pathoduet, possible weights are: "pathoduet_HE", "pathoduet_IHC".
            - For Phikon, Uni, and Conch, the weights are automatically downloaded from the Hugging Face model hub.

    Returns:
        A tuple containing a torch model and the transforms used to preprocess the images (set to None if no pretrained
            weights were used).
    """
    if arch_name not in arch_dict:
        raise ValueError(f"invalid model name. Possible models: {arch_dict.keys()}")
    model, pretrained_transforms = arch_dict[arch_name](weights=pretrained_weights)
    return model, pretrained_transforms


def compute_optimal_number_clusters(
    embeddings_matrix: np.ndarray,
    model_name: str,
    metric_name: str = "davies_bouldin",
    min_clusters: int = 2,
    max_clusters: int = 10,
    **kwargs,
) -> tuple[int, list[float]]:
    """Compute the optimal number of clusters to retrieve from the embedding matrix
    using the specified clustering model and metric.

    Args:
        embeddings_matrix: An array of shape (n_samples, n_features) containing the embeddings.
        model_name: The name of the clustering model. At the moment, possible models are:

            - "[kmeans](https://scikit-learn.org/stable/modules/generated/
            sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)"
            - "[kmeans_mini_batch](https://scikit-learn.org/stable/modules/generated/
            sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans)"

        metric_name: The name of the metric used to evaluate the clustering quality.
            At the moment, possible metrics are:

            - "[silhouette](https://scikit-learn.org/stable/modules/generated/
            sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score)"
            - "[calinski_harabasz](https://scikit-learn.org/stable/modules/generated/
            sklearn.metrics.calinski_harabasz_score.html#sklearn.metrics.calinski_harabasz_score)"
            - "[davies_bouldin](https://scikit-learn.org/stable/modules/generated/
            sklearn.metrics.davies_bouldin_score.html#sklearn.metrics.davies_bouldin_score)"

        min_clusters: The minimum number of clusters to consider.
        max_clusters: The maximum number of clusters to consider.

        **kwargs: Some additional arguments for the clustering model (see the documentation of the clustering model).

    Returns:
        A tuple containing the optimal number of clusters and the quality scores for each number of clusters.
    """
    if metric_name not in clustering_metrics.keys():
        raise ValueError(f"Metric {metric_name} not implemented")
    metric = clustering_metrics[metric_name]
    if model_name == "kmeans":
        cluster_model = skl_cluster.KMeans(**kwargs)
    elif model_name == "kmeans_mini_batch":
        cluster_model = skl_cluster.MiniBatchKMeans(**kwargs)
    else:
        raise ValueError(f"Model {model_name} not implemented")

    scores = []
    for n_clusters in tqdm(
        range(min_clusters, max_clusters + 1),
        desc="Computing clustering quality " "scores for each number of clusters",
    ):
        cluster_model.set_params(n_clusters=n_clusters)
        cluster_assignments = cluster_model.fit_predict(embeddings_matrix)
        scores.append(metric(embeddings_matrix, cluster_assignments))
    if metric_name == "davies_bouldin":
        optimal_number = np.argmin(scores) + min_clusters
    else:
        optimal_number = np.argmax(scores) + min_clusters
    return optimal_number, scores

def extract_stain_features(
    patches: torch.Tensor,
    conv_matrix_name: str = "HED") -> torch.Tensor:
    """Extract the stain features from the images.
    
    Args:
        imgs: A tensor of shape (n_samples, n_channels, height, width) containing the images.
        conv_matrix_name: The name of the conversion matrix. At the moment, possible values are:

            - "HED": Hematoxylin, Eosin and DAB.
            - "HD": Hematoxylin and DAB.
            - "HD_custom": Custom Hematoxylin and DAB matrix.
    
    Returns:
        A tensor of shape (n_samples, n_channels, height, width) containing the stain features.
    """
    conv_matrix = torch.tensor(retrieve_conv_matrix(conv_matrix_name), dtype=torch.float32)
    eps = torch.tensor([1e-6], dtype=torch.float32)
    patches = torch.maximum(patches, eps)
    log_adjust = torch.log(eps)
    stains = torch.einsum('bcij,cd->bdij', torch.log(patches) / log_adjust, conv_matrix) 
    stains = torch.maximum(stains, torch.tensor([0.0], dtype=torch.float32))
    if conv_matrix_name.startswith("HD"):
        stains = stains[:, :2, ...]
    stains_flattened = stains.view(stains.size(0), stains.size(1), -1)
    feats = torch.concatenate([stains.mean(dim=(2, 3)), stains.std(dim=(2, 3)), stains.amin(dim=(2, 3)),
                         stains.amax(dim=(2, 3)), stains_flattened.median(dim=-1).values,
                         stains_flattened.quantile(0.25, dim=-1), stains_flattened.quantile(0.75, dim=-1)], dim=1)
    return feats

def get_cells_in_patch(
    cells_df: gpd.GeoDataFrame,
    coord: torch.Tensor, 
    patch_size: int, 
    cell_classes: list[str]
) -> gpd.GeoDataFrame:
    """Get the cells whose centroids are within the patch defined by the input coordinates and size.
    
    Args:
        cells_df: A geodataframe containing the cells.
        coord: A tensor of shape (2,) containing the coordinates of the patch.
        patch_size: The size of the patch.
        cell_classes: A list of cell classes.
    
    Returns:
        A geodataframe containing the cells in the patch ordered by cell classes.
    """
    x, y = coord.numpy()
    patch_polygon = box(x, y, x + patch_size, y + patch_size, ccw=False)
    cells_in_patch =  cells_df.loc[cells_df.centroid.within(patch_polygon),]
    cells_in_patch_grouped = []
    for cell_class in cell_classes:
        if cell_class in cells_in_patch['classification'].values:
            cell_class_in_patch_df = cells_in_patch[cells_in_patch["classification"] == cell_class]
        else:
            cell_class_in_patch_df = gpd.GeoDataFrame(columns=cells_df.columns,
                                                        data=[pd.Series({col: None for col in cells_df.columns})])
            cell_class_in_patch_df["classification"] = cell_class
        cells_in_patch_grouped.append(cell_class_in_patch_df)
    result = pd.concat(cells_in_patch_grouped, ignore_index=True)
    return result

    
def get_cell_properties(cells_df: gpd.GeoDataFrame) -> tuple[list[float], list[str]]:
    """Get the properties of the cells in the input dataframe.

    Args:
        cells_df: A geodataframe containing the cells.

    Returns:
        A list of 7 features describing the number of cells and their average morphological properties.
    """
    if cells_df.geometry.isnull().all():
        return [0.0] * 7
    else:
        N_cells = len(cells_df)
        avg_cells_area = cells_df.area.mean()
        avg_cells_perimeter = cells_df.length.mean()
        avg_cells_compactness = ((4 * np.pi * cells_df.area) / (cells_df.length ** 2)).mean()
        avg_cells_roundness = ((4 * cells_df.area) / (cells_df.convex_hull.length ** 2)).mean()
        avg_cells_solidity = (cells_df.area / cells_df.convex_hull.area).mean()
        avg_cells_elongation = ((cells_df.bounds["maxx"] - cells_df.bounds["minx"]) / (cells_df.bounds["maxy"] - cells_df.bounds["miny"])).mean()
        return [N_cells, avg_cells_area, avg_cells_perimeter, avg_cells_compactness,
                avg_cells_roundness, avg_cells_solidity, avg_cells_elongation]
    

def compute_cell_features(cells_df_with_classes: gpd.GeoDataFrame) -> np.ndarray:
    """Compute the features of the cells from an input geodataframe for each class.

    Args:
        cells_df_with_classes: A geodataframe containing the cells and their classes.

    Returns:
        The cells based features for each class.
    """
    assert "classification" in cells_df_with_classes.columns, "The input geodataframe must contain a 'classification' column"
    cells_feats = np.ravel(cells_df_with_classes.groupby("classification", sort=False).apply(get_cell_properties, include_groups=False).to_list())
    return cells_feats
