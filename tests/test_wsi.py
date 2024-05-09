import os
import logging
import sys
import requests
import openslide
import tiffslide
import numpy as np
from PIL import Image
from pathlib import Path

from prismtoolbox import WSI
from _utils import check_pickle_file, check_geojson_file, check_h5_file

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


class TestWSI():

    @classmethod
    def setup_class(self) -> None:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.urls = ["https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs",
                     "https://openslide.cs.cmu.edu/download/openslide-testdata/Hamamatsu/OS-2.ndpi"]
        self.engines = ["openslide", "tiffslide"]
        self.ROIs = [[2064, 12192, 22128, 32208], [43392, 19072, 96128, 72192]]
        self.wsi_paths = []
        self.tissue_contours_folder = "./contours"
        self.patches_folder = "./patches"
        self.visualization_folder = "./visualize"
        Path(self.tissue_contours_folder).mkdir(parents=True, exist_ok=True)
        Path(self.patches_folder).mkdir(parents=True, exist_ok=True)
        Path(self.visualization_folder).mkdir(parents=True, exist_ok=True)

        Path("./slides").mkdir(parents=True, exist_ok=True)
        for url in self.urls:
            slide_path = f"./slides/{os.path.basename(url)}"
            if not os.path.exists(slide_path):
                response = requests.get(url)
                with open(slide_path, "wb") as f:
                    f.write(response.content)
            self.wsi_paths.append(slide_path)

        self.WSI_objects = [WSI(wsi_path, engine=engine) for wsi_path, engine in zip(self.wsi_paths, self.engines)]

    def test_read(self, subtests):
        expected_values = {
            "CMU-1": {"level_count": 3, "instance": openslide.OpenSlide},
            "OS-2": {"level_count": 10, "instance": tiffslide.TiffSlide}
        }
        for i in range(len(self.WSI_objects)):
            with subtests.test(msg=f"Test slide {os.path.basename(self.wsi_paths[i])}"):
                WSI_object = self.WSI_objects[i]
                slide = WSI_object.slide
                assert WSI_object.slide_name in expected_values
                expected = expected_values[WSI_object.slide_name]
                assert isinstance(slide, expected["instance"])
                assert slide.level_count == expected["level_count"]

    def test_create_thumbnail(self, subtests):
        for i in range(len(self.WSI_objects)):
            with subtests.test(msg=f"Test slide {os.path.basename(self.wsi_paths[i])}"):
                WSI_object = self.WSI_objects[i]
                level = len(WSI_object.level_downsamples) - 1
                img = WSI_object.create_thumbnail(level)
                assert isinstance(img, Image.Image)
                assert img.mode == "RGB"
                assert img.size == WSI_object.level_dimensions[level]

    def test_set_roi(self, subtests):
        for i in range(len(self.WSI_objects)):
            with subtests.test(msg=f"Test slide {os.path.basename(self.wsi_paths[i])}"):
                WSI_object = self.WSI_objects[i]
                ROI = self.ROIs[i]
                if ROI is not None:
                    WSI_object.set_roi(ROI)
                else:
                    ROI = [0, 0, WSI_object.level_dimensions[0][0], WSI_object.level_dimensions[0][1]]
                    WSI_object.set_roi(ROI)
                assert isinstance(WSI_object.ROI, np.ndarray)
                assert len(WSI_object.ROI) == 4
                assert WSI_object.level_dimensions[0][0] >= ROI[2]
                assert WSI_object.level_dimensions[0][1] >= ROI[3]
                assert WSI_object.ROI_height == ROI[3] - ROI[1]
                assert WSI_object.ROI_width == ROI[2] - ROI[0]

    def test_contour_tissue(self, subtests):
        params = {"window_avg": 30, "window_eng": 3, "thresh": 40, "area_min": 3e3}
        for i in range(len(self.WSI_objects)):
            with subtests.test(msg=f"Test slide {os.path.basename(self.wsi_paths[i])}"):
                WSI_object = self.WSI_objects[i]
                ROI = self.ROIs[i]
                if ROI is not None:
                    params["inside_roi"] = True
                    WSI_object.set_roi(ROI)
                else:
                    params["inside_roi"] = False
                WSI_object.detect_tissue(seg_level=len(WSI_object.level_dimensions) // 2, **params)
                assert len(WSI_object.tissue_contours) > 0
                WSI_object.save_tissue_contours(self.tissue_contours_folder)
                assert check_pickle_file(os.path.join(self.tissue_contours_folder, f"{WSI_object.slide_name}.pkl"))
                WSI_object.save_tissue_contours(self.tissue_contours_folder, file_format="geojson", label="contours")
                assert check_geojson_file(os.path.join(self.tissue_contours_folder, f"{WSI_object.slide_name}.geojson"))

    def test_visualize_WSI(self, subtests):
        for i in range(len(self.WSI_objects)):
            with subtests.test(msg=f"Test slide {os.path.basename(self.wsi_paths[i])}"):
                WSI_object = self.WSI_objects[i]
                ROI = self.ROIs[i]
                if ROI is not None:
                    WSI_object.set_roi(ROI)
                    crop_roi = True
                else:
                    crop_roi = False
                contour_path = os.path.join(self.tissue_contours_folder, f"{WSI_object.slide_name}.pkl")
                if os.path.exists(contour_path):
                    WSI_object.load_tissue_contours(contour_path)
                    view_slide_only = True
                else:
                    view_slide_only = False
                img = WSI_object.visualize(vis_level=len(WSI_object.level_dimensions) // 2,
                                           number_contours=True, view_slide_only=view_slide_only,
                                           crop_roi=crop_roi)
                assert isinstance(img, Image.Image)
                assert img.mode == "RGB"
                slide_dim = (WSI_object.ROI_width, WSI_object.ROI_height) if crop_roi else WSI_object.level_dimensions[
                    0]
                img_dim = np.ceil([slide_dim[0] / WSI_object.level_downsamples[len(WSI_object.level_downsamples) // 2],
                                   slide_dim[1] / WSI_object.level_downsamples[len(WSI_object.level_downsamples) // 2]])
                assert (img.size == img_dim).all()
                img.save(os.path.join(self.visualization_folder, f"{WSI_object.slide_name}_visualize.png"))

    def test_apply_pathologist_annotations(self):
        pass

    def test_extract_patch(self, subtests):
        params = {"patch_size": 256, "patch_level": 2,
                  "step_size": 256}
        for i in range(len(self.WSI_objects)):
            with subtests.test(msg=f"Test slide {os.path.basename(self.wsi_paths[i])}"):
                WSI_object = self.WSI_objects[i]
                ROI = self.ROIs[i]
                if ROI is not None:
                    WSI_object.set_roi(ROI)
                    if os.path.exists(os.path.join(self.tissue_contours_folder, f"{WSI_object.slide_name}.pkl")):
                        WSI_object.load_tissue_contours(os.path.join(self.tissue_contours_folder,
                                                                     f"{WSI_object.slide_name}.pkl"))
                        params["mode"] = "contours"
                        params["contours_mode"] = "four_pt_hard"
                    else:
                        params["mode"] = "roi"
                else:
                    params["mode"] = "all"
                WSI_object.extract_patches(**params)
                assert len(WSI_object.coords) > 0
                assert WSI_object.coords_attrs["patch_size"] == params["patch_size"]
                assert WSI_object.coords_attrs["patch_level"] == params["patch_level"]
                assert max(np.abs(WSI_object.coords[0] - WSI_object.coords[1])) >= int(params["step_size"] * \
                                                                                       WSI_object.level_downsamples[
                                                                                           params["patch_level"]])
                WSI_object.save_patches(self.patches_folder, file_format="h5")
                assert check_h5_file(os.path.join(self.patches_folder, f"{WSI_object.slide_name}.h5"))
                WSI_object.save_patches(self.patches_folder, file_format="geojson", label="patches", merge=True)
                assert check_geojson_file(os.path.join(self.patches_folder, f"{WSI_object.slide_name}.geojson"))

    def test_stitching_WSI(self, subtests):
        for i in range(len(self.WSI_objects)):
            with subtests.test(msg=f"Test slide {os.path.basename(self.wsi_paths[i])}"):
                WSI_object = self.WSI_objects[i]
                if os.path.exists(os.path.join(self.patches_folder, f"{WSI_object.slide_name}.h5")):
                    WSI_object.load_patches(os.path.join(self.patches_folder, f"{WSI_object.slide_name}.h5"))
                else:
                    continue
                ROI = self.ROIs[i]
                if ROI is not None:
                    WSI_object.set_roi(self.ROIs[i])
                    crop_roi = True
                else:
                    crop_roi = False
                img = WSI_object.stitch(vis_level=len(WSI_object.level_dimensions) // 2,
                                        draw_grid=True, crop_roi=crop_roi)
                assert isinstance(img, Image.Image)
                assert img.mode == "RGB"
                slide_dim = (WSI_object.ROI_width, WSI_object.ROI_height) if crop_roi else WSI_object.level_dimensions[
                    0]
                img_dim = np.ceil([slide_dim[0] / WSI_object.level_downsamples[len(WSI_object.level_downsamples) // 2],
                                   slide_dim[1] / WSI_object.level_downsamples[len(WSI_object.level_downsamples) // 2]])
                assert (img.size == img_dim).all()
                img.save(os.path.join(self.visualization_folder, f"{WSI_object.slide_name}_stitch.png"))
