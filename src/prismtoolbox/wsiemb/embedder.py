from __future__ import annotations

import numpy as np
import torch
import time
import os
import logging
from tqdm import tqdm

from prismtoolbox.utils.torch_utils import BaseSlideHandler, BasePatchHandler

from .emb_utils import create_model

log = logging.getLogger(__name__)


class SlideEmbedder(BaseSlideHandler):
    def __init__(
        self,
        slide_dir: str,
        slide_ext: str,
        arch_name: str,
        pretrained_weights: str,
        batch_size: int,
        num_workers: int,
        transforms_dict: dict[str, dict[str, any]] | None = None,
        device: str = "cuda",
        engine: str = "openslide",
        coords_dir: str | None = None,
        patch_size: int | None = None,
        patch_level: int | None = None,
        need_login: bool = False,
    ):
        """The SlideEmbedder class is used to extract embeddings from patches extracted direclty from the slides.

        Args:
            slide_dir: The directory containing the slides.
            slide_ext: The extension of the slides.
            arch_name: The name of the architecture to use. 
                See [create_model][prismtoolbox.wsiemb.emb_utils.create_model] for available architectures.
            pretrained_weights: The path to the pretrained weights or the name of the pretrained weights. See
                [create_model][prismtoolbox.wsiemb.emb_utils.create_model] for available weights for each architecture.
            batch_size: The batch size to use for the dataloader.
            num_workers: The number of workers to use for the dataloader.
            transforms_dict: The dictionary of transforms to use.
                See [create_transforms][prismtoolbox.utils.torch_utils.create_transforms]
                for more information. If None, the pretrained transforms will be used.
            device: The device to use for the model.
            engine: The engine to use for reading the slides.
            coords_dir: The directory containing the coordinates of the patches as hdf5 files. If None,
                the patch_size and patch_level must be provided.
            patch_size: The size of the patches. If None, it will be extracted from the hdf5 files.
            patch_level: The level of the patches. If None, it will be extracted from the hdf5 files.
            need_login: Whether to login to the HuggingFace Hub (for Uni and Conch models).
        
        Attributes:
            slide_dir: The directory containing the slides.
            slide_ext: The extension of the slides.
            batch_size: The batch size to use for the dataloader.
            num_workers: The number of workers to use for the dataloader.
            transforms_dict: The dictionary of transforms to use.
            device: The device to use for the model.
            engine: The engine to use for reading the slides.
            coords_dir: The directory containing the coordinates of the patches as hdf5 files.
            patch_size: The size of the patches.
            patch_level: The level of the patches.
            model: The model to use for creating the embeddings.
            pretrained_transforms: The transforms used for the pretrained model.
            slides_processed: The list that will contain the name of the slides that have been processed.
            embeddings: The list that will contain the extracted embeddings.
            
        """
        super(SlideEmbedder, self).__init__(
            slide_dir,
            slide_ext,
            batch_size,
            num_workers,
            transforms_dict,
            engine,
            coords_dir,
            patch_size,
            patch_level,
        )

        if need_login:
            from huggingface_hub import login

            login()

        self.device = device
        self.model, self.pretrained_transforms = create_model(
            arch_name,
            pretrained_weights
        )
        log.info(
            f"Model {arch_name} loaded with pretrained weights {pretrained_weights}."
        )
        self.model.eval()
        self.model.to(self.device)

        self.embeddings = []

    def extract_embeddings(
        self,
        slide_name: str, 
        coords: np.ndarray | None = None, 
        show_progress: bool = True
    ):
        """Extract embeddings from the patches of a slide.

        Args:
            slide_name: The name of the slide to extract the embeddings from (without the extension).
            coords: The coordinates of the patches to extract the embeddings from. 
                If None, the coordinates will be loaded from the hdf5 file located in coords_dir.
            show_progress: Whether to show the progress bar.
        """
        log.info(f"Extracting embeddings from the patches of {slide_name}.")
        dataset = self.create_dataset(slide_name, coords=coords)
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
        self.slides_processed.append(slide_name)
        log.info(f"Extracted {len(embeddings)} patches from {slide_name}.")
        self.embeddings.append(torch.cat(embeddings, dim=0))

    def save_embeddings(
        self, 
        output_directory: str,
        flush_memory: bool = True,
        format: str = "pt"
    ):
        """Save the extracted embeddings to the chosen format under slide_name.format.

        Args:
            output_directory: The directory where to save the embeddings.
            flush_memory: Whether to remove the embeddings from self.embeddings after saving.
            format: The format to save the embeddings in. Possible formats: ['pt', 'npy']
            
        """
        if format not in ["pt", "npy"]:
            raise ValueError("invalid format, possible formats: ['pt', 'npy']")
        for slide_name, embeddings in zip(self.slides_processed, self.embeddings):
            output_path = os.path.join(output_directory, f"{slide_name}.{format}")
            if format == "pt":
                torch.save(embeddings, output_path)
            elif format == "npy":
                embeddings = embeddings.numpy()
                np.save(output_path, embeddings)
            else:
                raise NotImplementedError
            log.info(f"Embeddings for slide {slide_name} saved at {output_path}.")
        if flush_memory:
            self.embeddings = []
            self.slides_processed = []
            log.info("Memory flushed.")


class PatchEmbedder(BasePatchHandler):
    def __init__(
        self,
        img_folder: str,
        arch_name: str,
        pretrained_weights: str,
        batch_size: int,
        num_workers: int,
        transforms_dict: dict[str, dict[str, any]] | None = None,
        device: str = "cuda",
        need_login: bool = False,
    ):
        """The PatchEmbedder class is used to extract embeddings from patches extracted as images in a folder.

        Args:
            img_folder: The directory containing the images of the patches.
            arch_name: The name of the architecture to use. 
                See [create_model][prismtoolbox.wsiemb.emb_utils.create_model] for available architectures.
            pretrained_weights: The path to the pretrained weights or the name of the pretrained weights. See
                [create_model][prismtoolbox.wsiemb.emb_utils.create_model] for available weights for each architecture.
            batch_size: The batch size to use for the dataloader.
            num_workers: The number of workers to use for the dataloader.
            transforms_dict: The dictionary of transforms to use.
                See [create_transforms][prismtoolbox.utils.torch_utils.create_transforms] for more information.
                If None, the pretrained transforms will be used.
            device: The device to use for the model.
            need_login: Whether to login to the HuggingFace Hub (for Uni and Conch models).
        
        Attributes:
            img_folder: The directory containing the images of the patches.
            batch_size: The batch size to use for the dataloader.
            num_workers: The number of workers to use for the dataloader.
            transforms_dict: The dictionary of transforms to use.
            device: The device to use for the model.
            model: The model to use for creating the embeddings.
            pretrained_transforms: The transforms used when pretraining the model.
            embeddings: The list that will contain the extracted embeddings.
            
        """
        super(PatchEmbedder).__init__(
            img_folder, batch_size, num_workers, transforms_dict
        )
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

        if transforms_dict is None:
            log.info("No transform provided, using pretrained transform.")
            self.transform = self.pretrained_transforms

        log.info(f"Transforms created: {self.transform}.")
        self.embeddings = []

    def extract_embeddings(self, show_progress: bool = True):
        """Extract embeddings from the images in the img_folder.

        Args:
            show_progress: Whether to show the progress bar.
        """
        log.info(f"Extracting embeddings from images in {self.img_folder}.")
        dataset = self.create_dataset()
        dataloader = self.create_dataloader(dataset)
        start_time = time.time()
        embeddings = []
        for imgs, folder_id in tqdm(
            dataloader,
            desc=f"Extracting embeddings from images in {self.img_folder}",
            disable=not show_progress,
        ):
            log.info(f"Extracting embeddings from folder {folder_id}.")
            imgs = imgs.to(self.device)
            with torch.no_grad():
                output = self.model(imgs)
                embeddings.append(output.cpu())
        log.info(f"Embedding time: {time.time() - start_time}.")
        log.info(f"Extracted {len(embeddings)} rom images in {self.img_folder}.")
        self.embeddings.append(embeddings)