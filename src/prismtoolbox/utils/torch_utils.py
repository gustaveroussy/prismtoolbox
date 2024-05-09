from __future__ import annotations

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transformsv2
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

from .data_utils import read_h5_file
from prismtoolbox import WSI
from prismtoolbox.utils.stain_utils import deconvolve_stain

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
    def __init__(self, min_value=0., max_value=1.):
        super(ClipCustom, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, img):
        img = img.clamp(min=self.min_value, max=self.max_value)
        img = (img - self.min_value) / (self.max_value - self.min_value)
        return img


class SlideDataset(Dataset):
    def __init__(self, coords, slide_path, patch_size, level, engine="openslide", transform=None,
                 deconvolve_channel=None):
        super(SlideDataset, self).__init__()
        self.coords = coords
        self.slide_path = slide_path
        self.patch_size = patch_size
        self.level = level
        self.engine = engine

        self.slide = WSI.read(self.slide_path, engine=self.engine)

        self.transform = ToTensorv2() if transform is None else transform

        self.deconvolve_channel = deconvolve_channel

    def worker_init(self, worker_id):
        self.slide = WSI.read(self.slide_path, engine=self.engine)
    
    def get_sample_patch(self):
        transform = ToTensorv2()
        random_idx = np.random.randint(len(self))
        patch = self.slide.read_region(self.coords[random_idx], self.level, (self.patch_size, self.patch_size)).convert("RGB")
        return transform(patch)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        patch = self.slide.read_region(coord, self.level, (self.patch_size, self.patch_size)).convert("RGB")
        if self.deconvolve_channel is not None:
            deconvolve_imgs = deconvolve_stain(patch)
            patch = Image.fromarray(deconvolve_imgs[self.deconvolve_channel])
        if self.transform:
            patch = self.transform(patch)
        return patch, coord


class BaseSlideHandler:
    def __init__(self, slide_dir, slide_ext, batch_size, num_workers, transforms_dict=None, engine="openslide",
                 coords_dir=None, patch_size=None, patch_level=None):
        self.slide_dir = slide_dir
        self.slide_ext = slide_ext
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms_dict = transforms_dict
        self.engine = engine
        self.coords_dir = coords_dir
        self.patch_size = patch_size
        self.patch_level = patch_level

        if (patch_size is None or patch_level is None) and coords_dir is None:
            raise ValueError("no patch size or patch level provided while no coords_dir provided. Please provide either"
                             "coords_dir or patch_size and patch_level.")

        self.slides_processed = []

    def create_dataset(self, slide_name, coords=None, deconvolve_channel=None):
        if self.transforms_dict is not None:
            log.info("Creating transform from transforms_dict.")
            transform = create_transforms(self.transforms_dict)
        elif self.pretrained_transforms is not None:
            log.info("No transform provided, using pretrained transform.")
            transform = self.pretrained_transforms
        else:
            log.info("No transform provided.")
            transform = None
        if coords is None:
            if self.coords_dir is None:
                raise ValueError(
                    "no coords provided and no coords_dir provided. Please provide either coords to this function or "
                    "coords_dir to the constructor of SlideEmbedder.")
            h5_path = os.path.join(self.coords_dir, f"{slide_name}.h5")
            coords, attrs = read_h5_file(h5_path, 'coords')
            log.info(f"Coords loaded from h5 file. Found {len(coords)} patches.")
        patch_size = self.patch_size if self.patch_size is not None else attrs['patch_size']
        patch_level = self.patch_level if self.patch_level is not None else attrs['patch_level']
        slide_path = os.path.join(self.slide_dir, f"{slide_name}.{self.slide_ext}")
        dataset = SlideDataset(coords, slide_path, patch_size, patch_level, self.engine, transform,
                               deconvolve_channel)
        log.info(f"Dataset created for {slide_name}.")
        return dataset

    def create_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, worker_init_fn=dataset.worker_init,
                          num_workers=self.num_workers)


class BasePatchHandler:
    def __init__(self, img_folder, batch_size, num_workers, transforms_dict=None):
        self.img_folder = img_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms_dict = transforms_dict

    def create_dataset(self):
        transform = create_transforms(self.transforms_dict) if self.transforms_dict is not None else None
        dataset = ImageFolder(self.img_folder, transform=transform)
        log.info(f"Created dataset from {self.img_folder} using ImageFolder from torchvision.")
        return dataset

    def create_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)


possible_transforms = {"totensor": ToTensorv2,
                       "clip_custom": ClipCustom,
                       "normalize": transformsv2.Normalize,
                       "horizontal_flip": transformsv2.RandomHorizontalFlip,
                       "vertical_flip": transformsv2.RandomVerticalFlip,
                       "rotation": transformsv2.RandomRotation,
                       "resized_crop": transformsv2.RandomResizedCrop,
                       "random_crop": transformsv2.RandomCrop,
                       "center_crop": transformsv2.CenterCrop,
                       "resize": transformsv2.Resize, }


def create_transforms(transform_dict: dict[str, dict[str, any]]) -> transformsv2.Compose:
    """Create a torchvision.transforms.Compose object from a dictionary of transforms.

    Args:
        transform_dict: Dictionary of transforms. The keys are the names of the transforms and the values are the
            parameters to pass to the transform as a dictionary. Possible transforms are:
            
            - "totensor": ToTensorv2
            - "normalize": transformsv2.Normalize
            - "horizontal_flip": transformsv2.RandomHorizontalFlip
            - "vertical_flip": transformsv2.RandomVerticalFlip
            - "rotation": transformsv2.RandomRotation
            - "resized_crop": transformsv2.RandomResizedCrop
            - "random_crop": transformsv2.RandomCrop
            - "center_crop": transformsv2.CenterCrop
            - "resize": transformsv2.Resize
            - "clip_custom": ClipCustom
                
            Please refer to the [torchvision documentation](https://pytorch.org/vision/stable/transforms.html) for the
            parameters of each torchvision transform.

    Returns:
        A torchvision.transforms.Compose object.
    """
    if any(transform_name not in possible_transforms for transform_name in transform_dict):
        raise ValueError(f"invalid transform name. Possible transforms: {possible_transforms.keys()}")
    transform = transformsv2.Compose([possible_transforms[transform_name](**transform_params) for transform_name,
    transform_params in transform_dict.items()])
    return transform

