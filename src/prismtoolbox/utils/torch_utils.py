from __future__ import annotations

import logging
import os
import inspect

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transformsv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from prismtoolbox import WSI
from prismtoolbox.utils.stain_utils import deconvolve_img

from .data_utils import read_h5_file

log = logging.getLogger(__name__)


class ToTensorv2(nn.Module):
    def __init__(self):
        super(ToTensorv2, self).__init__()
        self.toImage = transformsv2.ToImage()
        self.toDtype = transformsv2.ToDtype(torch.float32, scale=True)

    def forward(self, x):
        x = self.toImage(x)
        x = self.toDtype(x)
        return x


class ClipCustom(nn.Module):
    def __init__(self, min_value=0.0, max_value=1.0):
        super(ClipCustom, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, img):
        img = img.clamp(min=self.min_value, max=self.max_value)
        img = (img - self.min_value) / (self.max_value - self.min_value)
        return img


class SlideDataset(Dataset):
    def __init__(
        self,
        coords,
        slide_path,
        patch_size,
        level,
        downsample,
        engine="openslide",
        transforms=None,
        deconvolve_matrix=None,
        deconvolve_channel=None,
        coords_only=False,
    ):
        super(SlideDataset, self).__init__()
        self.coords = coords
        self.slide_path = slide_path
        self.patch_size = patch_size
        self.level = level
        self.downsample = downsample
        self.engine = engine

        WSI_object = WSI(self.slide_path, engine=self.engine)
        self.slide = WSI_object.slide
        self.slide_properties = WSI_object.properties
        self.slide_offset = WSI_object.offset

        self.transforms = ToTensorv2() if transforms is None else transforms

        self.deconvolve_matrix = deconvolve_matrix
        self.deconvolve_channel = deconvolve_channel
        self.coords_only = coords_only

    def worker_init(self, worker_id):
        self.slide = WSI.read(self.slide_path, engine=self.engine)

    def get_sample_patch(self):
        random_idx = np.random.randint(len(self))
        patch = self.slide.read_region(
            self.coords[random_idx], self.level, (self.patch_size, self.patch_size)
        ).convert("RGB")
        if self.transforms:
            patch = self.transforms(patch)
        return patch

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        if self.coords_only:
            return coord
        patch = self.slide.read_region(
            coord, self.level, (self.patch_size, self.patch_size)
        ).convert("RGB")
        if self.deconvolve_channel is not None and self.deconvolve_matrix is not None:
            stain_imgs = deconvolve_img(patch, self.deconvolve_matrix)
            patch = Image.fromarray(stain_imgs[self.deconvolve_channel])
        if self.transforms:
            patch = self.transforms(patch)
        return patch, coord


class BaseSlideHandler:
    def __init__(
        self,
        slide_dir,
        batch_size,
        num_workers,
        transforms_dict=None,
        engine="openslide",
        coords_dir=None,
        patch_size=None,
        patch_level=None,
        patch_downsample=None,
    ):
        self.slide_dir = slide_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms_dict = transforms_dict
        self.engine = engine
        self.coords_dir = coords_dir
        self.patch_size = patch_size
        self.patch_level = patch_level
        self.patch_downsample = patch_downsample

    def get_transforms(self):
        if self.transforms_dict is not None:
            log.info("Creating transforms from transforms dict.")
            transforms = create_transforms(self.transforms_dict)
        else:
            log.info("No transforms dict found.")
            transforms = ToTensorv2()
        return transforms

    def create_dataset(
        self,
        slide_name,
        slide_ext,
        coords=None,
        deconvolve_matrix=None,
        deconvolve_channel=None,
        coords_only=False,
        no_transforms=False,
    ):
        transforms = self.get_transforms() if not no_transforms else None
        if coords is None:
            if self.coords_dir is None:
                raise ValueError(
                    "no coords provided and no coords_dir provided. Please provide either coords to this function or "
                    "coords_dir to the constructor of SlideEmbedder."
                )
            h5_path = os.path.join(self.coords_dir, f"{slide_name}.h5")
            coords, attrs = read_h5_file(h5_path, "coords")
            log.info(f"Coords loaded from h5 file. Found {len(coords)} patches.")
        else:
            if (
                self.patch_size is None
                or self.patch_level is None
                or self.patch_downsample is None
            ):
                raise ValueError(
                    "no patch size or patch level or patch downsample provided. Please provide either "
                    "coords_dir or patch_size and patch_level and patch_downsample."
                )
        patch_size = (
            self.patch_size if self.patch_size is not None else attrs["patch_size"]
        )
        patch_level = (
            self.patch_level if self.patch_level is not None else attrs["patch_level"]
        )
        patch_downsample = (
            self.patch_downsample
            if self.patch_downsample is not None
            else attrs["downsample"]
        )
        slide_path = os.path.join(self.slide_dir, f"{slide_name}.{slide_ext}")
        dataset = SlideDataset(
            coords,
            slide_path,
            patch_size,
            patch_level,
            patch_downsample,
            self.engine,
            transforms,
            deconvolve_matrix,
            deconvolve_channel,
            coords_only,
        )
        log.info(f"Dataset created for {slide_name}, with transforms: {transforms}.")
        return dataset

    def create_dataloader(self, dataset, batch_size=None, num_workers=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        num_workers = self.num_workers if num_workers is None else num_workers
        return DataLoader(
            dataset,
            batch_size=batch_size,
            worker_init_fn=dataset.worker_init,
            num_workers=num_workers,
        )


class BasePatchHandler:
    def __init__(self, batch_size, num_workers, transforms_dict=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms_dict = transforms_dict

    def get_transforms(self):
        if self.transforms_dict is not None:
            log.info("Creating transforms from transforms dict.")
            transforms = create_transforms(self.transforms_dict)
        else:
            log.info("No transform dict found.")
            transforms = ToTensorv2()
        return transforms

    def create_dataset(self, img_folder):
        transforms = self.get_transforms()
        dataset = ImageFolder(img_folder, transform=transforms)
        log.info(
            f"Created dataset from {img_folder} using ImageFolder from torchvision, with transforms: {transforms}."
        )
        return dataset

    def create_dataloader(self, dataset, batch_size=None, num_workers=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

def get_torchvision_transforms() -> dict[str, transformsv2.Transform]:
    torchvision_transforms = {}
    #accepted_type = ["color", "geometry", "type_conversion"]
    to_remove = ["ToTensor"]
    for name, obj in inspect.getmembers(transformsv2):
        if inspect.isclass(obj) and issubclass(obj, transformsv2.Transform):
            if name not in to_remove:
                #if any(transform_type in inspect.getfile(obj) for transform_type in accepted_type):
                torchvision_transforms[name.lower()] = obj
    return torchvision_transforms

possible_transforms = {
    "totensor": ToTensorv2,
    "clip_custom": ClipCustom,
    **get_torchvision_transforms(),
}
    
def create_transforms(transforms_dict: dict[str, dict[str, any]]) -> transformsv2.Compose:
    """Create a torchvision.transforms.Compose object from a dictionary of transforms.

    Args:
        transforms_dict: Dictionary of transforms. The keys are the names of the transforms and the values are the
            parameters to pass to the transform as a dictionary. Possible transforms are:

            - "totensor": ToTensorv2. A custom transform that converts a PIL image to a tensor as done in torchvision's
                original ToTensor transform.
            - "clip_custom": ClipCustom.
            - Any torchvision v2 transform. The name of the transform should be in lowercase.

            Please refer to the [torchvision documentation](https://pytorch.org/vision/stable/transforms.html) for the
            parameters of each torchvision transform.

    Returns:
        A torchvision.transforms.Compose object.
    """
    if any(
        transform_name not in possible_transforms for transform_name in transforms_dict
    ):
        raise ValueError(
            f"invalid transform name. Possible transforms: {possible_transforms.keys()}"
        )
    transforms = transformsv2.Compose(
        [
            possible_transforms[transform_name](**transform_params)
            for transform_name, transform_params in transforms_dict.items()
        ]
    )
    return transforms
