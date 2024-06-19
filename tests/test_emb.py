import os
import sys
import logging
import requests
from pathlib import Path

import pytest
import numpy as np
from prismtoolbox import WSI
from prismtoolbox.wsiemb import SlideEmbedder, PatchEmbedder

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

class TestSlideEmbedder():
    
    @classmethod
    def setup_class(self) -> None:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.urls = ["https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs",
                     "https://openslide.cs.cmu.edu/download/openslide-testdata/Hamamatsu/OS-2.ndpi"]
        self.engine = "openslide"
        self.rng = np.random.default_rng(42)
        self.wsi_paths = []

        Path("./slides").mkdir(parents=True, exist_ok=True)
        for url in self.urls:
            slide_path = f"./slides/{os.path.basename(url)}"
            if not os.path.exists(slide_path):
                response = requests.get(url)
                with open(slide_path, "wb") as f:
                    f.write(response.content)
            self.wsi_paths.append(slide_path)
        
        # Simulate the extraction of coordinates
        
        self.coords_dir = "./coords"
        self.patches_dir = "./patches"
        self.wsi_coords = []
        Path(self.coords_dir).mkdir(parents=True, exist_ok=True)
        
        for slide_path in self.wsi_paths:
            WSI_object = WSI(slide_path, engine=self.engine)
            dimensions = WSI_object.dimensions
            coords = np.hstack([self.rng.choice(dimensions[0], size=(10,1)), self.rng.choice(dimensions[1], size=(10,1))])
            self.wsi_coords.append(coords)
            WSI_object.coords = coords
            coords_attr = {"patch_size": 256,
                           "patch_level": 0,
                           "downsample": WSI_object.level_downsamples[0],
                           "downsampled_level_dim": tuple(np.array(WSI_object.level_dimensions[0])),
                           "level_dim": WSI_object.level_dimensions[0],
                           "name": WSI_object.slide_name,}
            WSI_object.coords_attrs = coords_attr
            WSI_object.save_patches(self.coords_dir)
            WSI_object.save_patches(self.patches_dir, file_format="jpg")

        self.embeddings_dir = "./embeddings"
    
    @pytest.mark.parametrize("coord_dir,arch_name,pretrained_weights,transforms_dict,batch_size,num_workers,device,format", [
        (None,"clam", "IMAGENET1K_V2", {"totensor": {}, "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}, 4, 1, "cpu", "pt"),
        ("./coords", "phikon", None, None, 8, 2, "cpu", "npy")])
    def test_slide_embedder(
        self,
        coord_dir,
        arch_name, 
        pretrained_weights, 
        transforms_dict, 
        batch_size,
        num_workers,
        device,
        format,
        subtests
    ) -> None:
        
        if coord_dir is None:
            patch_level = 0
            patch_size = 256
        else:
            patch_level = None
            patch_size = None
        
        slide_embedder = SlideEmbedder(slide_dir="./slides", 
                                       coords_dir=coord_dir,
                                       patch_level=patch_level,
                                       patch_size=patch_size,
                                       arch_name=arch_name,
                                       pretrained_weights=pretrained_weights,
                                       transforms_dict=transforms_dict,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       device=device)

        for i, slide_path in enumerate(self.wsi_paths):
            slide_name, slide_ext = os.path.basename(slide_path).split(".")
            with subtests.test(msg=f"Test slide {slide_name}"):
                if coord_dir is None:
                    coords = self.wsi_coords[i]
                else:
                    coords = None
                # Extract the embeddings
                slide_embedder.extract_embeddings(slide_name, slide_ext=slide_ext, coords=coords)
                assert len(slide_embedder.embeddings[i]) == 10
                assert slide_name in slide_embedder.slides_processed
                if arch_name == "clam":
                    assert slide_embedder.embeddings[i].shape[-1] == 1024
                elif arch_name == "phikon":
                    assert slide_embedder.embeddings[i].shape[-1] == 768
                else:
                    raise ValueError(f"Unknown architecture name {arch_name}")
                # Save the embeddings
                slide_embedder.save_embeddings(self.embeddings_dir, flush_memory=False, format=format)
                assert os.path.exists(f"{self.embeddings_dir}/{slide_name}.{format}")
    
    def test_patch_embedder(self):
        pass
    
        