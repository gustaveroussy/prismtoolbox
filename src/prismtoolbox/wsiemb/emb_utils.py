from __future__ import annotations
import numpy as np
import torch.nn as nn
import torchvision.transforms.v2 as transformsv2
import sklearn.cluster as skl_cluster
from functools import partial
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
