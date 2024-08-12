import os
import sys
import logging
import requests
from pathlib import Path

import pytest
import numpy as np
from prismtoolbox import WSI
from prismtoolbox.wsiemb import SlideEmbedder, PatchEmbedder

from _utils import generate_fake_cell_segmentation, load_embeddings

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
                     "https://openslide.cs.cmu.edu/download/openslide-testdata/Hamamatsu/OS-3.ndpi"]
        self.engine = "openslide"
        self.rng = np.random.default_rng(42)
        self.n_samples_coords = 10
        self.n_cells_by_patch = 10
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
        self.cell_detection_dir = "./cell_detection"
        self.wsi_coords = []
        Path(self.coords_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cell_detection_dir).mkdir(parents=True, exist_ok=True)
        
        for slide_path in self.wsi_paths:
            WSI_object = WSI(slide_path, engine=self.engine)
            dimensions = WSI_object.dimensions
            coords = np.hstack([self.rng.choice(dimensions[0], size=(self.n_samples_coords,1)), self.rng.choice(dimensions[1], size=(10,1))])
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
            generate_fake_cell_segmentation(os.path.join(self.cell_detection_dir, f"{WSI_object.slide_name}.geojson"),
                                            coords, 
                                            256,
                                            ["tumor", "stroma"],
                                            self.n_cells_by_patch,
                                            WSI_object.offset)

        self.embeddings_dir = "./embeddings"
        self.slide_embedders = []
    
    @pytest.mark.parametrize("coord_dir,arch_name,pretrained_weights,transforms_dict,batch_size,num_workers,device", [
        (None,"clam", "IMAGENET1K_V2", {"totensor": {}, "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}, 1, 1, "cpu"),
        ("./coords", "phikon", None, None, 8, 2, "cpu")])
    def test_extract_model_based_embeddings(
        self,
        coord_dir,
        arch_name, 
        pretrained_weights, 
        transforms_dict, 
        batch_size,
        num_workers,
        device,
        subtests
    ) -> None:
        
        if coord_dir is None:
            patch_level = 0
            patch_size = 256
            patch_downsample = 1.0
        else:
            patch_level = None
            patch_size = None
            patch_downsample = None
        
        slide_embedder = SlideEmbedder(slide_dir="./slides", 
                                       coords_dir=coord_dir,
                                       patch_level=patch_level,
                                       patch_size=patch_size,
                                       patch_downsample=patch_downsample,
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
                slide_embedder.extract_model_based_embeddings(slide_name, slide_ext=slide_ext, coords=coords)
                assert slide_name in slide_embedder.model_based_embeddings.keys()
                assert len(slide_embedder.model_based_embeddings[slide_name]) == self.n_samples_coords
                if arch_name == "clam":
                    assert slide_embedder.model_based_embeddings[slide_name].shape[-1] == 1024
                    assert len(slide_embedder.model_based_embedding_names) == 1024
                elif arch_name == "phikon":
                    assert slide_embedder.model_based_embeddings[slide_name].shape[-1] == 768
                    assert len(slide_embedder.model_based_embedding_names) == 768
                else:
                    raise ValueError(f"Unknown architecture name {arch_name}")
            self.slide_embedders.append(slide_embedder)
    
    @pytest.mark.parametrize("coord_dir,batch_size, num_workers, device, conv_matrix_name", [
        (None, 1, 1, "cpu", "HED"), ("./coords", 2, 10, "cuda", "HD")])
    def test_extract_stain_based_embeddings(
        self,
        coord_dir,
        batch_size,
        num_workers,
        device,
        conv_matrix_name,
        subtests
    ):
        if coord_dir is None:
            patch_level = 0
            patch_size = 256
            patch_downsample = 1.0
        else:
            patch_level = None
            patch_size = None
            patch_downsample = None
        
        slide_embedder = SlideEmbedder(slide_dir="./slides", 
                                        coords_dir=coord_dir,
                                        patch_level=patch_level,
                                        patch_size=patch_size,
                                        patch_downsample=patch_downsample,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        device=device)
        
        if conv_matrix_name.startswith("HD"):
            nb_features = 14
        else:
            nb_features = 21
        for i, slide_path in enumerate(self.wsi_paths):
            slide_name, slide_ext = os.path.basename(slide_path).split(".")
            with subtests.test(msg=f"Test slide {slide_name}"):
                if coord_dir is None:
                    coords = self.wsi_coords[i]
                else:
                    coords = None
                # Extract the embeddings
                slide_embedder.extract_stain_based_embeddings(slide_name,
                                                              slide_ext=slide_ext,
                                                              coords=coords,
                                                              conv_matrix_name=conv_matrix_name)
                assert slide_name in slide_embedder.stain_based_embeddings.keys()
                assert len(slide_embedder.stain_based_embeddings[slide_name]) == 10
                assert slide_embedder.stain_based_embeddings[slide_name].shape[-1] == nb_features
                assert len(slide_embedder.stain_based_embedding_names) == nb_features
            if len(self.slide_embedders) == i:
                self.slide_embedders.append(slide_embedder)
            else:
                self.slide_embedders[i].stain_based_embeddings = slide_embedder.stain_based_embeddings
                self.slide_embedders[i].stain_based_embedding_names = slide_embedder.stain_based_embedding_names
    
    
    @pytest.mark.parametrize("coord_dir,batch_size, num_workers, device, cell_classes,with_offset", [
        (None, 1, 1, "cpu", ["tumor"], True), ("./coords", 2, 10, "cuda", ["tumor", "stroma"], True)])
    def test_extract_cell_based_embeddings(
        self, 
        coord_dir,
        batch_size,
        num_workers,
        device,
        cell_classes,
        with_offset,
        subtests
    ):
        if coord_dir is None:
            patch_level = 0
            patch_size = 256
            patch_downsample = 1.0
        else:
            patch_level = None
            patch_size = None
            patch_downsample = None
        
        slide_embedder = SlideEmbedder(slide_dir="./slides", 
                                        coords_dir=coord_dir,
                                        patch_level=patch_level,
                                        patch_size=patch_size,
                                        patch_downsample=patch_downsample,
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
                slide_embedder.extract_cell_based_embeddings(slide_name,
                                                            slide_ext=slide_ext,
                                                            coords=coords,
                                                            cells_path=f"{self.cell_detection_dir}/{slide_name}.geojson",
                                                            cell_classes=cell_classes,
                                                            with_offset=with_offset)
                assert slide_name in slide_embedder.cell_based_embeddings.keys()
                assert len(slide_embedder.cell_based_embeddings[slide_name]) == 10
                assert slide_embedder.cell_based_embeddings[slide_name].shape[-1] == 7 * len(cell_classes)
                assert (slide_embedder.cell_based_embeddings[slide_name][:, 0::7] == self.n_cells_by_patch).all()
                assert len(slide_embedder.cell_based_embedding_names) == 7 * len(cell_classes)
            if len(self.slide_embedders) == i:
                self.slide_embedders.append(slide_embedder)
            else:
                self.slide_embedders[i].cell_based_embeddings = slide_embedder.cell_based_embeddings
                self.slide_embedders[i].cell_based_embedding_names = slide_embedder.cell_based_embedding_names
    
    @pytest.mark.parametrize("format,merge,", [("pt", True), ("npy", False)])
    def test_save_embeddings(self, format, merge, subtests):
         for i, slide_path in enumerate(self.wsi_paths):
            slide_name, _ = os.path.basename(slide_path).split(".")
            with subtests.test(msg=f"Test slide {slide_name}"):
                slide_embedder = self.slide_embedders[i]
                slide_embedder.save_embeddings(self.embeddings_dir, format=format, merge=merge)
                if merge:
                    assert os.path.exists(f"{self.embeddings_dir}/{slide_name}.{format}")
                    emb = load_embeddings(f"{self.embeddings_dir}/{slide_name}.{format}", format)
                    assert len(emb) == self.n_samples_coords
                else:
                    emb_files = []
                    for suffix in ["model_based", "cell_based", "stain_based"]:
                        assert os.path.exists(f"{self.embeddings_dir}/{slide_name}_{suffix}.{format}")
                        emb_files.append(load_embeddings(f"{self.embeddings_dir}/{slide_name}_{suffix}.{format}", format))
                    assert all([len(emb) == self.n_samples_coords for emb in emb_files])    