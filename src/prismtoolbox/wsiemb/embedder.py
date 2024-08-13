from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from prismtoolbox.utils.data_utils import read_json_with_geopandas
from prismtoolbox.utils.torch_utils import BasePatchHandler, BaseSlideHandler

from .emb_utils import (
    CELL_FEATURE_NAMES,
    STAIN_FEATURE_NAMES,
    compute_cell_features,
    create_model,
    extract_stain_features,
    get_cells_in_patch,
)

log = logging.getLogger(__name__)


class SlideEmbedder(BaseSlideHandler):
    def __init__(
        self,
        slide_dir: str,
        batch_size: int,
        num_workers: int,
        arch_name: str | None = None,
        pretrained_weights: str | None = None,
        transforms_dict: dict[str, dict[str, any]] | None = None,
        device: str = "cuda",
        engine: str = "openslide",
        coords_dir: str | None = None,
        patch_size: int | None = None,
        patch_level: int | None = None,
        patch_downsample: int | None = None,
        need_login: bool = False,
    ):
        """The SlideEmbedder class is used to extract embeddings from patches extracted direclty from the slides.

        Args:
            slide_dir: The directory containing the slides.
            batch_size: The batch size to use for the dataloader.
            num_workers: The number of workers to use for the dataloader.
            arch_name: The name of the architecture to use.
                See [create_model][prismtoolbox.wsiemb.emb_utils.create_model] for available architectures.
            pretrained_weights: The path to the pretrained weights or the name of the pretrained weights. See
                [create_model][prismtoolbox.wsiemb.emb_utils.create_model] for available weights for each architecture.
            transforms_dict: The dictionary of transforms to use.
                See [create_transforms][prismtoolbox.utils.torch_utils.create_transforms]
                for more information. If None, the pretrained transforms will be used.
            device: The device to use for the model.
            engine: The engine to use for reading the slides.
            coords_dir: The directory containing the coordinates of the patches as hdf5 files. If None,
                the patch_size and patch_level must be provided.
            patch_size: The size of the patches. If None, it will be extracted from the hdf5 files.
            patch_level: The level of the patches. If None, it will be extracted from the hdf5 files.
            patch_downsample: The downsample of the patches. If None, it will be extracted from the hdf5 files.
            need_login: Whether to login to the HuggingFace Hub (for Uni and Conch models).

        Attributes:
            slide_dir: The directory containing the slides.
            batch_size: The batch size to use for the dataloader.
            num_workers: The number of workers to use for the dataloader.
            transforms_dict: The dictionary of transforms to use.
            device: The device to use for the model.
            engine: The engine to use for reading the slides.
            coords_dir: The directory containing the coordinates of the patches as hdf5 files.
            patch_size: The size of the patches.
            patch_level: The level of the patches.
            patch_downsample: The downsample of the patches.
            arch_name: The name of the architecture to use.
            model: The model to use for creating the embeddings.
            pretrained_transforms: The transforms used for the pretrained model.
            model_based_embeddings: A dictionary containing the extracted embeddings for each slide
                with the pretrained model.
            stain_based_embeddings: A dictionary containing the extracted embeddings for each slide
                with the stain based features.
            cell_based_embeddings: A dictionary containing the extracted embeddings for each slide
                with the cell based features.
            model_based_embedding_names: The names of the features extracted with the pretrained model.
            stain_based_embedding_names: The names of the features extracted with the stain based features.
            cell_based_embedding_names: The names of the features extracted with the cell based features.

        """
        super().__init__(
            slide_dir,
            batch_size,
            num_workers,
            transforms_dict,
            engine,
            coords_dir,
            patch_size,
            patch_level,
            patch_downsample,
        )

        if need_login:
            from huggingface_hub import login

            login()

        self.device = device
        self.arch_name = arch_name
        if self.arch_name is not None:
            self.model, self.pretrained_transforms = create_model(
                arch_name, pretrained_weights
            )
            log.info(
                f"Model {self.arch_name} loaded with pretrained weights {pretrained_weights}."
            )
            self.model.eval()
            self.model.to(self.device)
        else:
            self.model = None
            self.pretrained_transforms = None

        self.model_based_embeddings = {}
        self.stain_based_embeddings = {}
        self.cell_based_embeddings = {}
        self.model_based_embedding_names = []
        self.stain_based_embedding_names = []
        self.cell_based_embedding_names = []

    def get_transforms(self):
        """Get the transforms to use for creating the dataset.

        Returns:
            The transforms to use when loading the patches.
        """
        if self.transforms_dict is not None:
            transforms = super().get_transforms()
        elif self.pretrained_transforms is not None:
            log.info("No transforms dict found, using pretrained transforms.")
            transforms = self.pretrained_transforms
        else:
            log.info("No transforms dict or pretrained transforms found.")
            transforms = None
        return transforms

    def extract_model_based_embeddings(
        self,
        slide_name: str,
        slide_ext: str,
        coords: np.ndarray | None = None,
        show_progress: bool = True,
    ):
        """Extract embeddings from the patches of a slide using the pretrained model.

        Args:
            slide_name: The name of the slide to extract the embeddings from (without the extension).
            slide_ext: The extension of the slide.
            coords: The coordinates of the patches to extract the embeddings from.
                If None, the coordinates will be loaded from the hdf5 file located in coords_dir.
            show_progress: Whether to show the progress bar.
        """
        assert (
            self.model is not None
        ), "Model not found. Please provide an architecture name when initializing the SlideEmbedder."
        log.info(f"Extracting embeddings from the patches of {slide_name}.")
        dataset = self.create_dataset(slide_name, slide_ext=slide_ext, coords=coords)
        dataloader = self.create_dataloader(dataset)
        start_time = time.time()
        embeddings = []
        for patches, _ in tqdm(
            dataloader,
            desc=f"Extracting embeddings from the patches of {slide_name}",
            disable=not show_progress,
        ):
            patches = patches.to(self.device)
            with torch.no_grad():
                output = self.model(patches)
                embeddings.append(output.cpu())
        log.info(f"Embedding time: {time.time() - start_time}.")
        self.model_based_embeddings[slide_name] = torch.cat(embeddings, dim=0)
        embeddings_dim = self.model_based_embeddings[slide_name].shape[1]
        self.model_based_embedding_names = [
            f"{self.arch_name}_{i}" for i in range(embeddings_dim)
        ]
        log.info(f"Extracted embeddings from {len(embeddings)} patches of {slide_name}.")

    def extract_stain_based_embeddings(
        self,
        slide_name: str,
        slide_ext: str,
        coords: np.ndarray | None = None,
        conv_matrix_name: str = "HED",
        show_progress: bool = True,
    ):
        """Extract embeddings from the patches of a slide using the stain based features.

        Args:
            slide_name: The name of the slide to extract the embeddings from (without the extension).
            slide_ext: The extension of the slide.
            coords: The coordinates of the patches to extract the embeddings from.
                If None, the coordinates will be loaded from the hdf5 file located in coords_dir.
            conv_matrix_name: The name of the convolutional matrix to use for the stain based features. See
                [extract_stain_features][prismtoolbox.wsiemb.emb_utils.extract_stain_features] for available matrices.
            show_progress: Whether to show the progress bar.
        """
        log.info(f"Extracting embeddings from the patches of {slide_name}.")
        dataset = self.create_dataset(
            slide_name, slide_ext=slide_ext, coords=coords, no_transforms=True
        )
        dataloader = self.create_dataloader(dataset)
        if conv_matrix_name == "HED":
            stain_names = ["Hematoxylin", "Eosin", "DAB"]
        else:
            stain_names = ["Hematoxylin", "DAB"]
        self.stain_based_embedding_names = [
            f"{stain_name}_{feature}"
            for feature in STAIN_FEATURE_NAMES
            for stain_name in stain_names
        ]
        start_time = time.time()
        embeddings = []
        for patches, _ in tqdm(
            dataloader,
            desc=f"Extracting stain based features from the patches of {slide_name}",
            disable=not show_progress,
        ):
            embeddings.append(extract_stain_features(patches, conv_matrix_name))
        log.info(f"Embedding time: {time.time() - start_time}.")
        self.stain_based_embeddings[slide_name] = torch.cat(embeddings, dim=0)
        log.info(f"Extracted embeddings from {len(embeddings)} patches of {slide_name}.")

    def extract_cell_based_embeddings(
        self,
        slide_name: str,
        slide_ext: str,
        coords: np.ndarray | None = None,
        cells_path: str | None = None,
        cell_classes: list[str] | None = None,
        with_offset: bool = True,
        show_progress: bool = True,
    ):
        """Extract embeddings from the patches of a slide using the cell based features.

        Args:
            slide_name: The name of the slide to extract the embeddings from (without the extension).
            slide_ext: The extension of the slide.
            coords: The coordinates of the patches to extract the embeddings from.
                If None, the coordinates will be loaded from the hdf5 file located in coords_dir.
            cells_path: The path to the cells geojson file.
            cell_classes: The classes of the cells to extract the embeddings from. If None, all the classes will be used.
            with_offset: Whether to offset the coordinates of the cells by the slide offset.
            show_progress: Whether to show the progress bar.
        """
        dataset = self.create_dataset(
            slide_name, slide_ext=slide_ext, coords=coords, coords_only=True
        )
        dataloader = self.create_dataloader(dataset, num_workers=1)
        patch_size = dataset.patch_size
        patch_downsample = dataset.downsample
        ref_patch_size = patch_size * patch_downsample
        offset = dataset.slide_offset if with_offset else (0, 0)
        offset = (-offset[0], -offset[1])
        cells_df = read_json_with_geopandas(cells_path, offset=offset)
        if "classification" not in cells_df.columns:
            raise ValueError(
                "The 'classification' column is missing in the cells dataframe."
            )
        cell_classes = (
            cells_df.classification.unique() if cell_classes is None else cell_classes
        )
        self.cell_based_embedding_names = [
            f"{cell_class}_{feature}"
            for cell_class in cell_classes
            for feature in CELL_FEATURE_NAMES
        ]
        start_time = time.time()
        embeddings = []
        if self.num_workers > 1:
            pool = mp.Pool(self.num_workers)
        for coords in tqdm(
            dataloader,
            desc=f"Extracting cell based features from the patches of {slide_name}",
            disable=not show_progress,
        ):
            if self.num_workers > 1:
                cells_df_in_patches = pool.starmap(
                    get_cells_in_patch,
                    [(cells_df, coord, ref_patch_size, cell_classes) for coord in coords],
                )
                cells_features = np.array(
                    pool.map(compute_cell_features, cells_df_in_patches)
                )
            else:
                cells_df_in_patches = [
                    get_cells_in_patch(cells_df, coord, ref_patch_size, cell_classes)
                    for coord in coords
                ]
                cells_features = np.array(
                    [
                        compute_cell_features(cells_df_in_patch)
                        for cells_df_in_patch in cells_df_in_patches
                    ]
                )
            embeddings.append(torch.tensor(cells_features))
        if self.num_workers > 1:
            pool.close()
        self.cell_based_embeddings[slide_name] = torch.cat(embeddings, dim=0)
        log.info(f"Embedding time: {time.time() - start_time}.")
        log.info(f"Extracted embeddings from {len(embeddings)} patches of {slide_name}.")

    def save_embeddings(
        self,
        save_dir: str,
        flush_memory: bool = False,
        format: str = "pt",
        merge: bool = False,
    ):
        """Save the extracted embeddings to the chosen format under slide_name.format.

        Args:
            save_dir: The path to the directory where to save the embeddings.
            flush_memory: Whether to remove the embeddings from self.embeddings after saving.
            format: The format to save the embeddings in. Possible formats: ['pt', 'npy']
            merge: Whether to merge the different types of embeddings before saving.

        """
        if not os.path.isdir(save_dir):
            log.warning(f"Folder {save_dir} does not exist, creating new folder...")
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        if format not in ["pt", "npy"]:
            raise ValueError("invalid format, possible formats: ['pt', 'npy']")
        slides_processed = set(self.model_based_embeddings.keys()).union(
            self.stain_based_embeddings.keys(), self.cell_based_embeddings.keys()
        )
        for slide_name in slides_processed:
            if merge:
                embeddings_list = [
                    (
                        "",
                        torch.cat(
                            [
                                emb_dict[slide_name]
                                for emb_dict in [
                                    self.model_based_embeddings,
                                    self.stain_based_embeddings,
                                    self.cell_based_embeddings,
                                ]
                                if slide_name in emb_dict.keys()
                            ],
                            dim=1,
                        ),
                    )
                ]
            else:
                embeddings_list = [
                    (emb_type, emb_dict[slide_name])
                    for emb_type, emb_dict in zip(
                        ["_model_based", "_stain_based", "_cell_based"],
                        [
                            self.model_based_embeddings,
                            self.stain_based_embeddings,
                            self.cell_based_embeddings,
                        ],
                    )
                    if slide_name in emb_dict.keys()
                ]
            for emb_type, emb in embeddings_list:
                output_path = os.path.join(save_dir, f"{slide_name+emb_type}.{format}")
                if format == "pt":
                    torch.save(emb, output_path)
                elif format == "npy":
                    emb = emb.numpy()
                    np.save(output_path, emb)
                else:
                    raise NotImplementedError
            log.info(f"Embeddings for slide {slide_name} saved at {output_path}.")

        if flush_memory:
            self.model_based_embeddings = {}
            self.stain_based_embeddings = {}
            self.cell_based_embeddings = {}
            log.info("Memory flushed.")

    def save_embeddings_names(
        self,
        save_dir: str,
        format: str = "csv",
        flush_memory: bool = False,
        merge: bool = False,
    ):
        """Save the extracted embeddings names to the chosen format under slide_name.format.

        Args:
            save_dir: The path to the directory where to save the embeddings.
            format: The format to save the embeddings in. Possible formats: ['csv']
            flush_memory: Whether to remove the reset the embeddings names after saving.
            merge: Whether the embeddings were merged before saving.
        """
        if not os.path.isdir(save_dir):
            log.warning(f"Folder {save_dir} does not exist, creating new folder...")
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        if format not in ["csv"]:
            raise ValueError("invalid format, possible formats: ['csv']")

        if merge:
            output_path = os.path.join(save_dir, f"embeddings_names.{format}")
            if os.path.exists(output_path):
                log.warning(
                    f"File embeddings_names.{format} already exists in {save_dir}. Overwriting..."
                )
            embeddings_names = (
                self.model_based_embedding_names
                + self.stain_based_embedding_names
                + self.cell_based_embedding_names
            )
            df = pd.DataFrame(embeddings_names, columns=["feature_names"])
            df.to_csv(output_path, index=False)
        else:
            for emb_type, emb_names in zip(
                ["model_based", "stain_based", "cell_based"],
                [
                    self.model_based_embedding_names,
                    self.stain_based_embedding_names,
                    self.cell_based_embedding_names,
                ],
            ):
                if len(emb_names) == 0:
                    continue
                output_path = os.path.join(
                    save_dir, f"embeddings_names_{emb_type}.{format}"
                )
                if os.path.exists(output_path):
                    log.warning(
                        f"File embeddings_names_{emb_type}.{format} already exists in {save_dir}. Overwriting..."
                    )
                df = pd.DataFrame(emb_names, columns=["feature_names"])
                df.to_csv(output_path, index=False)
        log.info(f"Embeddings names saved at {output_path}.")
        if flush_memory:
            self.model_based_embedding_names = []
            self.stain_based_embedding_names = []
            self.cell_based_embedding_names = []
            log.info("Memory flushed.")


class PatchEmbedder(BasePatchHandler):
    def __init__(
        self,
        arch_name: str,
        batch_size: int,
        num_workers: int,
        pretrained_weights: str | None = None,
        transforms_dict: dict[str, dict[str, any]] | None = None,
        device: str = "cuda",
        need_login: bool = False,
    ):
        """The PatchEmbedder class is used to extract embeddings from patches extracted as images in a folder.

        Args:
            arch_name: The name of the architecture to use.
                See [create_model][prismtoolbox.wsiemb.emb_utils.create_model] for available architectures.
            batch_size: The batch size to use for the dataloader.
            num_workers: The number of workers to use for the dataloader.
            pretrained_weights: The path to the pretrained weights or the name of the pretrained weights. See
                [create_model][prismtoolbox.wsiemb.emb_utils.create_model] for available weights for each architecture.
            transforms_dict: The dictionary of transforms to use.
                See [create_transforms][prismtoolbox.utils.torch_utils.create_transforms] for more information.
                If None, the pretrained transforms will be used.
            device: The device to use for the model.
            need_login: Whether to login to the HuggingFace Hub (for Uni and Conch models).

        Attributes:
            batch_size: The batch size to use for the dataloader.
            num_workers: The number of workers to use for the dataloader.
            transforms_dict: The dictionary of transforms to use.
            device: The device to use for the model.
            model: The model to use for creating the embeddings.
            pretrained_transforms: The transforms used when pretraining the model.
            embeddings: A dictionary containing the extracted embeddings for each image.

        """
        super().__init__(batch_size, num_workers, transforms_dict)
        self.device = device

        if need_login:
            from huggingface_hub import login

            login()

        self.model, self.pretrained_transforms = create_model(
            arch_name, pretrained_weights
        )
        log.info(
            f"Model {arch_name} loaded with pretrained weights {pretrained_weights}."
        )
        self.model.eval()
        self.model.to(self.device)

        self.embeddings = {}

    def get_transforms(self):
        """Get the transforms to use for creating the embeddings.

        Returns:
            The transforms to use when loading the patches.
        """
        if self.transforms_dict is not None:
            super().get_transforms()
        elif self.pretrained_transforms is not None:
            log.info("No transforms dict found, using pretrained transforms.")
            transforms = self.pretrained_transforms
        else:
            log.info("No transforms dict or pretrained transforms found.")
            transforms = None
        return transforms

    def extract_embeddings(self, img_folder, show_progress: bool = True):
        """Extract embeddings from the images in the img_folder.

        Args:
            img_folder: A folder containing a series of subfolders, each containing images.
                For example, img_folder could be a folder where the subfolders correspond to different slides.
            show_progress: Whether to show the progress bar.
        """
        log.info(f"Extracting embeddings from images in {img_folder}.")
        dataset = self.create_dataset(img_folder=img_folder)
        dataloader = self.create_dataloader(dataset)
        start_time = time.time()
        embeddings = [[] for _ in range(len(dataset.classes))]
        img_ids = []
        for i in range(len(dataset.classes)):
            img_ids.append(np.array(dataset.imgs)[np.array(dataset.targets) == i][:, 0])
        for imgs, folder_id in tqdm(
            dataloader,
            desc=f"Extracting embeddings from images in {img_folder}",
            disable=not show_progress,
        ):
            imgs = imgs.to(self.device)
            with torch.no_grad():
                output = self.model(imgs)
                for i in range(len(dataset.classes)):
                    embeddings[i].append(output[folder_id == i].cpu())
        log.info(f"Embedding time: {time.time() - start_time}.")
        log.info(f"Extracted {len(embeddings)} from images in {img_folder}.")
        for img_id, embedding in zip(img_ids, embeddings):
            self.embeddings[img_id] = torch.cat(embedding, dim=0)
