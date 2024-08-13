import logging
import multiprocessing as mp
import os
import time

import cv2
import ipywidgets as widgets
import numpy as np
from IPython.display import display
from PIL import Image
from tqdm import tqdm

from prismtoolbox import WSI
from prismtoolbox.utils.qupath_utils import (
    contoursToPolygons,
    export_polygons_to_qupath,
)
from prismtoolbox.utils.torch_utils import BaseSlideHandler, ClipCustom
from prismtoolbox.wsicore.core_utils import contour_mask

from .seg_utils import create_segmentation_tools, solve_conflicts

log = logging.getLogger(__name__)


class NucleiSegmenter(BaseSlideHandler):
    def __init__(
        self,
        slide_dir,
        model_name,
        pretrained_weights,
        batch_size,
        num_workers,
        transforms_dict=None,
        device="cuda",
        engine="openslide",
        coords_dir=None,
        patch_size=None,
        patch_level=None,
        deconvolve_channel=None,
        deconvolve_matrix="HE",
        threshold_overlap=0.3,
        **kwargs_seg_tool,
    ):
        super().__init__(
            slide_dir,
            batch_size,
            num_workers,
            transforms_dict,
            engine,
            coords_dir,
            patch_size,
            patch_level,
        )

        (
            self.preprocessing_fct,
            self.segmentation_fct,
            self.postprocessing_fct,
        ) = create_segmentation_tools(
            model_name, pretrained_weights, device, **kwargs_seg_tool
        )
        self.deconvolve_channel = deconvolve_channel
        self.deconvole_matrix = deconvolve_matrix
        self.threshold_overlap = threshold_overlap
        self.nuclei_seg = {}

    def set_clip_custom_params_on_ex(
        self, slide_name, slide_ext, coords=None, deconvolve_channel=None
    ):
        dataset = self.create_dataset(
            slide_name,
            slide_ext=slide_ext,
            coords=coords,
            deconvolve_channel=deconvolve_channel,
        )
        sample_patch = dataset.get_sample_patch()

        def apply_clip_custom(min_value, max_value):
            clip_custom = ClipCustom(min_value, max_value)
            result_tensor = clip_custom(sample_patch).permute(1, 2, 0)
            result_image = Image.fromarray((result_tensor.numpy() * 255).astype("uint8"))
            display(result_image)

        w = widgets.interact(
            apply_clip_custom,
            min_value=widgets.FloatSlider(min=0, max=0.5, step=0.01, value=0.1),
            max_value=widgets.FloatSlider(min=0.5, max=1.0, step=0.01, value=0.9),
        )
        display(w)

    def segment_nuclei(
        self,
        slide_name,
        slide_ext,
        deconvolve_channel=None,
        coords=None,
        merge=False,
        show_progress=True,
    ):
        log.info(f"Extracting embeddings from the patches of {slide_name}.")
        dataset = self.create_dataset(
            slide_name,
            slide_ext=slide_ext,
            coords=coords,
            deconvolve_channel=deconvolve_channel,
        )
        dataloader = self.create_dataloader(dataset)
        start_time = time.time()
        masks = []
        for patches, coord in tqdm(
            dataloader,
            desc=f"Extracting nuclei from the patches of {slide_name}",
            disable=not show_progress,
        ):
            if self.preprocessing_fct:
                patches = self.preprocessing_fct(patches)
            output = self.segmentation_fct(patches)
            if self.postprocessing_fct:
                output = self.postprocessing_fct(output)
            masks.extend(zip(coord.numpy(), output))
        log.info(f"Embedding time: {time.time() - start_time}.")
        nuclei = self.process_masks(masks, merge)
        self.nuclei_seg[slide_name] = nuclei
        log.info(f"Extracted {len(nuclei.geoms)} patches from {slide_name}.")

    @staticmethod
    def _process_mask(coord, labeled_mask, border_width=3, min_point_cnt=5):
        nuclei = []
        for nucleus_idx in np.unique(labeled_mask):
            if nucleus_idx == 0:
                continue

            canvas = (labeled_mask == nucleus_idx).astype("uint8")

            border_mask = np.pad(
                np.zeros(
                    (
                        canvas.shape[0] - 2 * border_width,
                        canvas.shape[1] - 2 * border_width,
                    )
                ),
                border_width,
                constant_values=1,
            )

            if np.any((canvas * border_mask) != 0):
                continue

            contours = contour_mask(canvas)

            if len(contours) > 1:
                contours_nucleus = cv2.convexHull(np.concatenate(contours))
            else:
                contours_nucleus = contours[0]

            if len(contours_nucleus) < min_point_cnt:
                continue
            else:
                nuclei.append(contours_nucleus + coord)

        return nuclei

    def process_masks(self, masks, merge=False):
        pool = mp.Pool(self.num_workers)
        patch_nuclei = pool.starmap(self._process_mask, masks)
        pool.close()
        pool.join()
        nuclei = [nucleus for nuclei in patch_nuclei for nucleus in nuclei]
        nuclei = contoursToPolygons(nuclei, merge)
        nuclei = solve_conflicts(nuclei.geoms, self.threshold_overlap)
        return nuclei

    def save_nuclei(self, output_directory, slide_ext, flush_memory=True):
        for slide_name, nuclei in self.nuclei_seg.items():
            WSI_object = WSI(os.path.join(self.slide_dir, f"{slide_name}.{slide_ext}"))
            offset = WSI_object.offset
            output_path = os.path.join(output_directory, f"{slide_name}.geojson")
            export_polygons_to_qupath(
                nuclei,
                output_path,
                "detection",
                offset,
                color=(255, 0, 0),
            )
            log.info(f"Embeddings for slide {slide_name} saved at {output_path}.")
        if flush_memory:
            self.nuclei_seg = []
            self.slides_processed = []
            log.info("Memory flushed.")
