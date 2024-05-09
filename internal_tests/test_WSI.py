import os
import openslide
import tiffslide
import cv2
import numpy as np
from PIL import Image
import mock
import builtins
from pathlib import Path
from unittest import TestCase

from src.prismtoolbox import WSI


class TestWSI(TestCase):

    def setUp(self) -> None:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.wsi_paths = ["/media/l_le-bescond/DATA/Documents/Daisy_10_03_22/data/16HC017295.mrxs",
                          "/media/l_le-bescond/DATA/Documents/brca/TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF"
                          "-D8FB-45CF-B4BD-C6B76294C291.svs"]
        self.engines = ["openslide", "tiffslide"]
        self.ROIs = [(1856, 82432, 53888, 126144), None]
        self.tissue_contours_folder = "./contours_test"
        self.visualization_folder = "./visualize_test"
        self.WSI_objects = [WSI(wsi_path, engine=engine) for wsi_path, engine in zip(self.wsi_paths, self.engines)]

    def test_read(self):
        for i in range(len(self.WSI_objects)):
            with self.subTest(slide=i):
                WSI_object = self.WSI_objects[i]
                slide = WSI_object.slide
                if WSI_object.engine == "openslide":
                    self.assertIsInstance(slide, openslide.OpenSlide)
                    self.assertEqual(slide.level_count, 9)
                elif WSI_object.engine == "tiffslide":
                    self.assertIsInstance(slide, tiffslide.TiffSlide)
                    self.assertEqual(slide.level_count, 4)

    def test_create_thumbnail(self):
        for i in range(len(self.WSI_objects)):
            with self.subTest(msg=f"Test slide {os.path.basename(self.wsi_paths[i])}"):
                WSI_object = self.WSI_objects[i]
                level = len(WSI_object.level_downsamples) - 1
                img = WSI_object.create_thumbnail(level)
                self.assertIsInstance(img, Image.Image)
                self.assertEqual(img.mode, "RGB")
                self.assertEqual(img.size, WSI_object.level_dimensions[level])
                
    @mock.patch.object(builtins, 'input', lambda _: "")
    def test_set_roi(self):
        for i in range(len(self.WSI_objects)):
            with self.subTest(msg=f"Test slide {os.path.basename(self.wsi_paths[i])}"):
                WSI_object = self.WSI_objects[i]
                level_to_view = len(WSI_object.level_downsamples) - 1
                if i == 0:
                    WSI_object.set_roi()
                    img = WSI_object.create_thumbnail(level_to_view)
                    cv2.imshow("Test cropped ROI - press ESC to quit", np.array(img.crop(WSI_object.ROI /
                                                                                         WSI_object.level_downsamples[level_to_view])))
                    while True:
                        k = cv2.waitKey(0) & 0xFF
                        if k == 27:
                            cv2.destroyAllWindows()
                            break
                    print(f'ROI: {WSI_object.ROI}')
                    self.assertIsInstance(WSI_object.ROI, np.ndarray)
                    self.assertEqual(len(WSI_object.ROI), 4)
                    self.assertGreaterEqual(WSI_object.ROI_width, 50000)
                    self.assertLessEqual(WSI_object.ROI_height, 85000)
                else:
                    ROI = [0, 0, WSI_object.level_dimensions[0][0], WSI_object.level_dimensions[0][1]]
                    WSI_object.set_roi(ROI)
                    self.assertEqual(WSI_object.ROI.tolist(), ROI)
                    self.assertEqual(len(WSI_object.ROI), 4)

    def test_contour_tissue(self):
        params = {"seg_level": 3, "window_avg": 30, "window_eng": 3, "thresh": 40, "area_min": 3e3}
        for i in range(len(self.WSI_objects)):
            with self.subTest(msg=f"Test slide {os.path.basename(self.wsi_paths[i])}"):
                WSI_object = self.WSI_objects[i]
                ROI = self.ROIs[i]
                if ROI is not None:
                    params["inside_roi"] = True
                    WSI_object.set_roi(ROI)
                else:
                    params["inside_roi"] = False
                WSI_object.detect_tissue(**params)
                self.assertNotEqual(len(WSI_object.tissue_contours), 0)
                WSI_object.save_tissue_contours("./contours_test")
                if i==0:
                    WSI_object.save_tissue_contours("./contours_test", file_format="geojson")

    def test_visualize_WSI(self):
        params = {"vis_level": 3, "number_contours": True, "view_slide_only": False}
        for i in range(len(self.WSI_objects)):
            with self.subTest(msg=f"Test slide {os.path.basename(self.wsi_paths[i])}"):
                WSI_object = self.WSI_objects[i]
                ROI = self.ROIs[i]
                if ROI is not None:
                    WSI_object.set_roi(ROI)
                    params["crop_roi"] = True
                else:
                    params["crop_roi"] = False
                WSI_object.load_tissue_contours(os.path.join(self.tissue_contours_folder,
                                                             f"{WSI_object.slide_name}.pkl"))
                img = WSI_object.visualize(**params)
                save_folder = os.path.join(self.visualization_folder, "test_visualize_WSI")
                Path(save_folder).mkdir(parents=True, exist_ok=True)
                img.save(os.path.join(save_folder, f"{WSI_object.slide_name}.png"))

    def test_apply_pathologist_annotations(self):
        params_vis = {"vis_level": 3, "number_contours": True, "view_slide_only": False}
        path = "/media/l_le-bescond/DATA/Documents/Daisy_10_03_22/contours_pathologist/16HC017295.geojson"
        WSI_object = self.WSI_objects[0]
        WSI_object.load_tissue_contours(os.path.join(self.tissue_contours_folder,
                                                     f"{WSI_object.slide_name}.pkl"))
        WSI_object.apply_pathologist_annotations(path)
        print(len(WSI_object.tissue_contours))
        self.assertNotEqual(len(WSI_object.tissue_contours), 0)
        WSI_object.save_tissue_contours("./contours_test_pathologist_annotations", file_format="geojson")
        ROI = self.ROIs[0]
        if ROI is not None:
            WSI_object.set_roi(ROI)
            params_vis["crop_roi"] = True
        else:
            params_vis["crop_roi"] = False
        img = WSI_object.visualize(**params_vis)
        save_folder = os.path.join(self.visualization_folder, "test_visualize_WSI")
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        img.save(os.path.join(save_folder, f"{WSI_object.slide_name}_with_pathologist_annotations.png"))


    def test_extract_patch(self):
        params = {"patch_size": 256, "patch_level": 2,
                  "step_size": 256}
        for i in range(len(self.WSI_objects)):
            with self.subTest(msg=f"Test slide {os.path.basename(self.wsi_paths[i])}"):
                WSI_object = self.WSI_objects[i]
                ROI = self.ROIs[i]
                if ROI is not None:
                    WSI_object.set_roi(ROI)
                    WSI_object.load_tissue_contours(os.path.join(self.tissue_contours_folder,
                                                                 f"{WSI_object.slide_name}.pkl"))
                    params["mode"] = "contours"
                    params["contours_mode"] = "four_pt_hard"
                else:
                    params["mode"] = "all"
                WSI_object.extract_patches(**params)
                
    def test_stitching_WSI(self):
        save_folder = os.path.join(self.visualization_folder, "test_stitch_WSI")
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        for i in range(len(self.WSI_objects)):
            with self.subTest(msg=f"Test slide {os.path.basename(self.wsi_paths[i])}"):
                WSI_object = self.WSI_objects[i]
                WSI_object.load_patches(os.path.join("./patches_test", f"{WSI_object.slide_name}.h5"))
                ROI = self.ROIs[i]
                if ROI is not None:
                    WSI_object.set_roi(self.ROIs[i])
                    crop_roi = True
                else:
                    crop_roi = False
                selected_idx, colors = None, None
                #selected_idx = np.arange(10)
                #colors = np.array([(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                #          (0, 255, 255), (125, 125, 125), (0, 125, 125), (125, 0, 125)])
                img = WSI_object.stitch(vis_level=3, selected_idx=selected_idx, colors=colors, draw_grid=True,
                                               crop_roi=crop_roi)
                img.save(os.path.join(save_folder, f"{WSI_object.slide_name}.png"))
    



