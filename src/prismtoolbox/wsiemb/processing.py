from __future__ import annotations

import logging
import torch
import numpy as np
import umap
import umap.plot
import sklearn.cluster as skl_cluster
import sklearn.decomposition as skl_decomposition
import sklearn.manifold as skl_manifold

from prismtoolbox import WSI
from prismtoolbox.utils.data_utils import save_obj_with_pickle, load_obj_with_pickle
from prismtoolbox.utils.vis_utils import get_colors_from_cmap, plot_scatter
from .emb_utils import compute_optimal_number_clusters

log = logging.getLogger(__name__)


class EmbeddingProcessor:
    def __init__(
        self,
        embeddings: list[np.ndarray | torch.Tensor | str],
        slide_ids: list[str] | None = None,
        slide_labels: list[int] | None = None,
        cmap: str = "Set1",
        seed: int = None
    ):
        """_summary_

        Args:
            embeddings: The embeddings to process. Can be a list of numpy arrays, torch tensors or paths to embeddings.
            slide_ids: The ids of the slides corresponding to the embeddings.
            slide_labels: The labels of the slides corresponding to the embeddings.
            cmap: The colormap to use for visualizations.
            seed: The seed to use for reproducibility.
            
        Attributes:
            embeddings: The embeddings to process as a list of numpy arrays.
            embeddings_matrix: The embeddings to process concatenated into a single numpy array.
            embeddings_stats: The statistics of the embeddings (mean, std, min, max).
            slide_ids: The ids of the slides corresponding to the embeddings. If not provided,
                the slide ids are generated.
            slide_labels: The labels of the slides corresponding to the embeddings. If not provided,
                the slide labels are the slide ids.
            labels_matrix: The labels of the slides corresponding to the embeddings concatenated into a single
                numpy array.
            cmap: The colormap to use for visualizations.
            seed: The seed to use for reproducibility.
            cluster_model: The clustering model used to cluster the embeddings. It is set using the
            [create_cluster_model][prismtoolbox.wsiemb.processing.EmbeddingProcessor.create_cluster_model] method.
            n_clusters: The number of clusters in the clustering model. It is set using the
            [create_cluster_model][prismtoolbox.wsiemb.processing.EmbeddingProcessor.create_cluster_model] method.
            cluster_colors: The colors of the clusters in the clustering model. It is set using the
            [create_cluster_model][prismtoolbox.wsiemb.processing.EmbeddingProcessor.create_cluster_model] method.
        """
        self.embeddings = self.load_embeddings(embeddings)
        self.embeddings_matrix = np.concatenate(self.embeddings, axis=0)
        self.embeddings_stats = self.compute_embeddings_stats(self.embeddings_matrix)
        self.slide_ids = (
            np.array(slide_ids) if slide_ids is not None else np.arange(len(embeddings))
        )
        self.slide_labels = (
            np.array(slide_labels) if slide_labels is not None else slide_ids
        )
        self.labels_matrix = np.concatenate(
            [
                np.repeat(label, len(emb))
                for label, emb in zip(self.slide_labels, self.embeddings)
            ],
            axis=0,
        )
        self.cmap = cmap
        self.seed = seed
        self.cluster_model = None
        self.n_clusters = None
        self.cluster_colors = None

    @staticmethod
    def load_embeddings(embeddings: list[np.ndarray | torch.Tensor | str]) -> list[np.ndarray]:
        """Process the embeddings to load them as numpy arrays.

        Args:
            embeddings: The embeddings to process. Can be a list of numpy arrays, torch tensors or paths to embeddings.
            
        Returns:
            The embeddings loaded as numpy arrays.
        """
        if isinstance(embeddings, list):
            embeddings_loaded = []
            for emb in embeddings:
                if isinstance(emb, torch.Tensor):
                    embeddings_loaded.append(emb.numpy())
                elif isinstance(emb, np.ndarray):
                    embeddings_loaded.append(emb)
                elif isinstance(emb, str):
                    embeddings_loaded.append(torch.load(emb).numpy())
                else:
                    raise ValueError("embedding type not supported")
            return embeddings_loaded
        else:
            raise ValueError(
                "embedding type not supported, please provide a list of embeddings "
                "or a list of paths to embeddings"
            )

    @staticmethod
    def compute_embeddings_stats(embeddings_matrix: np.ndarray) -> dict[str, np.ndarray]:
        """Compute the statistics of an input embeddings matrix.

        Args:
            embeddings_matrix: The embeddings matrix to compute the statistics of.

        Returns:
            The statistics of the embeddings matrix (mean, std, min, max) as a dictionary.
        """
        return {
            "mean": np.mean(embeddings_matrix, axis=0),
            "std": np.std(embeddings_matrix, axis=0),
            "min": np.min(embeddings_matrix, axis=0),
            "max": np.max(embeddings_matrix, axis=0),
        }

    def return_subsampled_embeddings(
        self, 
        n_samples: int | float, 
        by_slide: bool = True, 
        with_labels: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Creates a subsample version of the embeddings matrix.

        Args:
            n_samples: The number of samples to subsample. If float, it is the percentage of samples to subsample.
            by_slide: Whether to subsample by slide or not.
            with_labels: Whether to subsample also the labels matrix or not.

        Returns:
            The subsampled embeddings matrix. If with_labels, it also returns the subsampled labels matrix.
        """
        if n_samples > len(self.embeddings_matrix):
            log.error(
                f"Number of samples {n_samples} is greater than the number of embeddings, returning all embeddings."
            )
            return self.embeddings_matrix
        rng = np.random.default_rng(self.seed)
        if by_slide:
            subsampled_idx = []
            current_idx = 0
            for emb in self.embeddings:
                if n_samples < 1:
                    n_samples_per_slide = int(n_samples * len(emb))
                else:
                    n_samples_per_slide = n_samples // len(self.embeddings)
                subsampled_idx.extend(
                    rng.choice(len(emb), n_samples_per_slide, replace=False) + current_idx
                )
                current_idx += len(emb)
            log.info(f"Subsampled {n_samples} embeddings from each slide.")
        else:
            if n_samples < 1:
                n_samples = int(n_samples * len(self.embeddings_matrix))
            subsampled_idx = rng.choice(
                len(self.embeddings_matrix), n_samples, replace=False
            )
            log.info(f"Subsampled {n_samples} embeddings.")
        if with_labels:
            return (
                self.embeddings_matrix[subsampled_idx],
                self.labels_matrix[subsampled_idx],
            )
        else:
            return self.embeddings_matrix[subsampled_idx]

    def get_embedding_for_slide(
        self,
        slide_id: str,
        normalize: bool = True
    ) -> np.ndarray:
        """Get the embeddings for a specific slide.

        Args:
            slide_id: The id of the slide to get the embeddings for.
            normalize: Whether to normalize the embeddings or not according to the mean and std of self.embeddings_stats.

        Returns:
            The embeddings for the slide.
        """
        if slide_id not in self.slide_ids:
            raise ValueError(f"slide {slide_id} not found in slide ids")
        idx = np.where(self.slide_ids == slide_id)[0].item()
        emb_mean, emb_std = self.embeddings_stats["mean"], self.embeddings_stats[
            "std"
        ] if normalize else (0, 1)
        return (self.embeddings[idx] - emb_mean) / emb_std

    def get_optimal_number_clusters(
        self,
        model_name: str,
        normalize: bool = True,
        metric_name: str = "davies_bouldin",
        min_clusters: int = 2,
        max_clusters: int = 10,
        with_scores: bool = True,
        n_samples: int | float | None = None,
        **kwargs
    )-> int | tuple[int, list[float]]:
        """Compute the optimal number of clusters for the embeddings.

        Args:
            model_name: The clustering model to use. 
                See [compute_optimal_number_clusters][prismtoolbox.wsiemb.emb_utils.compute_optimal_number_clusters] 
                for the available models.
            normalize: Whether to normalize the embeddings or not according to the mean and std of self.embeddings_stats.
            metric_name: The metric to use to compute the optimal number of clusters.
                See [compute_optimal_number_clusters][prismtoolbox.wsiemb.emb_utils.compute_optimal_number_clusters]
                for the available metrics.
            min_clusters: The minimum number of clusters to consider.
            max_clusters: The maximum number of clusters to consider.
            with_scores: Whether to return the scores or not.
            n_samples: The number of samples to subsample. If None, the whole embeddings matrix is used.
            **kwargs: Additional arguments for the clustering model (see the documentation of the clustering model).

        Returns:
            The optimal number of clusters according to the metric. If with_scores, it also returns the scores.
        """
        emb_mean, emb_std = self.embeddings_stats["mean"], self.embeddings_stats[
            "std"
        ] if normalize else (0, 1)
        if n_samples is not None:
            embeddings_matrix = self.return_subsampled_embeddings(n_samples)
        else:
            embeddings_matrix = self.embeddings_matrix
        embeddings_matrix = (embeddings_matrix - emb_mean) / emb_std
        optimal_number, scores = compute_optimal_number_clusters(
            embeddings_matrix,
            model_name,
            metric_name=metric_name,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            **kwargs,
        )
        if with_scores:
            return optimal_number, scores
        else:
            return optimal_number

    def create_cluster_model(
        self, 
        model_name: str, 
        normalize: bool = True, 
        n_samples: int | float | None =None, 
        **kwargs
    ):
        """Create a clustering model trained on the embeddings matrix. The resulting model is stored in
            self.cluster_model.

        Args:
            model_name: The clustering model to use. Possible models are:
                
                - "[kmeans](https://scikit-learn.org/stable/modules/generated/
                sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)"
                - "[kmeans_mini_batch](https://scikit-learn.org/stable/modules/generated/
                sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans)"
            
            normalize: Whether to normalize the embeddings or not according to the mean and std of
                self.embeddings_stats.
            n_samples: The number of samples to subsample. If None, the whole embeddings matrix is used.

        """
        if model_name == "kmeans":
            self.cluster_model = skl_cluster.KMeans(**kwargs)
        elif model_name == "kmeans_mini_batch":
            self.cluster_model = skl_cluster.MiniBatchKMeans(**kwargs)
        else:
            raise ValueError(f"model {model_name} not implemented")
        emb_mean, emb_std = self.embeddings_stats["mean"], self.embeddings_stats[
            "std"
        ] if normalize else (0, 1)
        if n_samples is not None:
            embeddings_matrix = self.return_subsampled_embeddings(n_samples)
        else:
            embeddings_matrix = self.embeddings_matrix
        embeddings_matrix = (embeddings_matrix - emb_mean) / emb_std
        self.cluster_model.fit(embeddings_matrix)
        self.n_clusters = self.cluster_model.n_clusters
        self.cluster_colors = get_colors_from_cmap(self.cmap, self.n_clusters)

    def get_cluster_assignments_for_slide(
        self, 
        slide_id: str, 
        normalize: bool = True
    ) -> np.ndarray:
        """Get the cluster assignments for a specific slide. Requires a cluster model to be created first
            with the [create_cluster_model][prismtoolbox.wsiemb.processing.EmbeddingProcessor.create_cluster_model]
            method.

        Args:
            slide_id: The id of the slide to get the cluster assignments for.
            normalize: Whether the embeddings were normalized or not when creating the cluster model.

        Returns:
            The cluster assignments for the slide.
        """
        assert (
            self.cluster_model is not None
        ), "no cluster model created, please create a cluster model first"
        if slide_id not in self.slide_ids:
            raise ValueError(f"slide {slide_id} not found in slide ids")
        idx = np.where(self.slide_ids == slide_id)[0].item()
        embeddings = self.embeddings[idx]
        emb_mean, emb_std = self.embeddings_stats["mean"], self.embeddings_stats[
            "std"
        ] if normalize else (0, 1)
        embeddings = (embeddings - emb_mean) / emb_std
        return self.cluster_model.predict(embeddings)

    def get_cluster_percentages_for_slide(
        self, 
        slide_id: str, 
        normalize: bool = True
    ) -> dict[str, float]:
        """Get the cluster percentages for a specific slide. Requires a cluster model to be created first
            with the [create_cluster_model][prismtoolbox.wsiemb.processing.EmbeddingProcessor.create_cluster_model]
            method.

        Args:
            slide_id: The id of the slide to get the cluster percentages for.
            normalize: Whether the embeddings were normalized or not when creating the cluster model.

        Returns:
            The cluster percentages for the slide.
        """
        cluster_assignments = self.get_cluster_assignments_for_slide(slide_id, normalize)
        cluster_percentage = {}
        for cluster in range(self.n_clusters):
            cluster_percentage[f"cluster_{cluster}"] = (
                cluster_assignments == cluster
            ).sum() / len(cluster_assignments)
        return cluster_percentage

    def export_clusters_to_qupath(
        self,
        WSI_object: WSI,
        save_dir: str,
    ):
        """Export the clusters as polygons to a geojson file for a slide.

        Args:
            WSI_object: An instance of the WSI class created from the slide.
            save_dir: The directory to save the geojson file to.
        """
        assert (
            self.n_clusters is not None
        ), "no cluster model created, please create a cluster model first"
        cluster_assignments = self.get_cluster_assignments_for_slide(
            WSI_object.slide_name
        )
        idx = np.arange(len(WSI_object.coords))
        assert len(cluster_assignments) == len(
            idx
        ), "Number of cluster assignments and number of patches do not match"
        for cluster in range(self.n_clusters):
            WSI_object.save_patches(
                save_dir,
                file_format="geojson",
                selected_idx=idx[cluster_assignments == cluster],
                merge=True,
                label=f"cluster_{cluster}",
                color=self.cluster_colors[cluster].tolist(),
                append_to_existing_file=True,
            )

    @staticmethod
    def scale_to_01_range(x: np.ndarray) -> np.ndarray:
        """Scale an array to the [0; 1] range.

        Args:
            x: The array to scale.

        Returns:
            The array scaled to the [0; 1] range.
        """
        # compute the distribution range
        value_range = np.max(x) - np.min(x)

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    def visualize(
        self,
        model_name: str,
        labels: np.ndarray | None = None,
        n_samples: int | float | None = None,
        **kwargs
    ):
        """Visualize the embeddings using a dimensionality reduction model.

        Args:
            model_name: The name of the dimensionality reduction model to use. Possible models are:
                
                - "[PCA](https://scikit-learn.org/stable/modules/generated/
                sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)"
                - "[TSNE](https://scikit-learn.org/stable/modules/generated/
                sklearn.manifold.TSNE.html#sklearn.manifold.TSNE)"
                - "[UMAP](https://umap-learn.readthedocs.io/en/latest/)"
            labels: The labels to use for the visualization. If None, the slide labels are used.
            n_samples: The number of samples to subsample. If None, the whole embeddings matrix is used.
        """
        if model_name == "PCA":
            dimensionality_reduction_model = skl_decomposition.PCA(
                n_components=2, **kwargs
            )
        elif model_name == "TSNE":
            dimensionality_reduction_model = skl_manifold.TSNE(n_components=2, **kwargs)
        elif model_name == "UMAP":
            dimensionality_reduction_model = umap.UMAP(n_components=2, **kwargs)
        else:
            raise ValueError(f"model {model_name} not implemented")
        if n_samples is not None:
            embeddings_matrix, slide_labels = self.return_subsampled_embeddings(
                n_samples, with_labels=True
            )
        else:
            embeddings_matrix = self.embeddings_matrix
            slide_labels = self.labels_matrix
        labels_plot = slide_labels if labels is None else labels
        embeddings_reduced = dimensionality_reduction_model.fit_transform(
            embeddings_matrix
        )
        if model_name == "UMAP":
            umap.plot.points(dimensionality_reduction_model, labels=labels_plot)
        else:
            plot_scatter(
                self.scale_to_01_range(embeddings_reduced[:, 0]),
                self.scale_to_01_range(embeddings_reduced[:, 1]),
                labels_plot,
                self.cmap,
            )

    def save_cluster_model(self, output_path: str):
        """Save the clustering model to a pickle file.

        Args:
            output_path: The path to save the clustering model to.
        """
        save_obj_with_pickle(self.cluster_model, output_path)

    def import_cluster_model(self, input_path: str):
        """Import the clustering model from a pickle file.

        Args:
            input_path: The path to import the clustering model from.
        """
        self.cluster_model = load_obj_with_pickle(input_path)
        self.n_clusters = self.cluster_model.n_clusters
        self.cluster_colors = get_colors_from_cmap(self.cmap, self.n_clusters)
