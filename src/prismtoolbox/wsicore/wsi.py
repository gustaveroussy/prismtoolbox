from __future__ import annotations

import logging

import os
import re
import warnings
import pathlib
import multiprocessing as mp

import pandas as pd
import cv2
import openslide
import tiffslide
import numpy as np
from PIL import Image

from .core_utils import (
    select_roi_on_thumbnail,
    local_average,
    compute_law_feats,
    apply_bilateral_filter,
    apply_binary_thresh,
    floodfill_img,
    contour_mask,
    IsInContour,
    isBlackPatch,
    isWhitePatch,
    save_patches_with_hdf5,
)

from prismtoolbox.utils.vis_utils import init_image

from prismtoolbox.utils.data_utils import (
    save_obj_with_pickle,
    load_obj_with_pickle,
    read_h5_file,
)
from prismtoolbox.utils.qupath_utils import (
    contoursToPolygons,
    PolygonsToContours,
    patchesToPolygons,
    read_qupath_annotations,
    export_polygons_to_qupath,
    intersectionPolygons,
)

log = logging.getLogger(__name__)


class WSI:
    def __init__(self, slide_path: str, engine: str = "openslide"):
        """The WSI (Whole Slide Image) class is responsible for handling operations
        related to whole slide images.

        Args:
            slide_path: The path to the slide image file.
            engine: The engine used to read the slide image.

        Attributes:
            slide_path (str): The path to the slide image file.
            engine (str): The engine used to read the slide image.
            slide_name (str): The name of the slide image file.
                Retrieved from the slide path using
                [retrieve_slide_name_ext][prismtoolbox.wsicore.WSI.retrieve_slide_name_ext] method.
            slide_ext (str): The extension of the slide image file.
                Retrieved from the slide path using
                [retrieve_slide_name_ext][prismtoolbox.wsicore.WSI.retrieve_slide_name_ext] method.
            slide (OpenSlide | TiffSlide): The wsi read from the file using engine.
            dimensions (list[tuple[int, int]]): The dimensions of the slide image.
                Set by the set_slide_attributes method.
            level_dimensions (list[tuple[int, int]]): The dimensions of the different levels of the slide image.
                Set by the [set_slide_attributes][prismtoolbox.wsicore.WSI.set_slide_attributes] method.
            level_downsamples (list[tuple[int, int]]): The downsampling factors of the different levels of
                the slide image. Set by the [set_slide_attributes][prismtoolbox.wsicore.WSI.set_slide_attributes]
                method.
            properties (dict): The properties of the slide image.
                Set by the [set_slide_attributes][prismtoolbox.wsicore.WSI.set_slide_attributes] method.
            offset (tuple[int, int]): The offset of the slide image.
                Set by the [set_slide_attributes][prismtoolbox.wsicore.WSI.set_slide_attributes] method.
            ROI (ndarray): The region of interest in the slide image.
                Please use the [set_roi][prismtoolbox.wsicore.WSI.set_roi] method to set the ROI.
            ROI_width (int): The width of the region of interest.
                Set by the [set_roi][prismtoolbox.wsicore.WSI.set_roi] method.
            ROI_height (int): The height of the region of interest.
                Set by the [set_roi][prismtoolbox.wsicore.WSI.set_roi] method.
            tissue_contours (list[ndarray]): The contours of the tissue in the slide image.
                Please use the [detect_tissue][prismtoolbox.wsicore.WSI.detect_tissue]
                method to detect the tissue contours.
            coords (np.ndarray): The coordinates of patches extracted from slide image.
                Please use the [extract_patches][prismtoolbox.wsicore.WSI.extract_patches]
                method to extract patches.
            coords_attrs (dict): The attributes of the coordinates.
                Set by the [extract_patches][prismtoolbox.wsicore.WSI.extract_patches] method.
        """
        self.slide_path = slide_path
        self.engine = engine
        self.slide_name, self.slide_ext = self.retrieve_slide_name_ext(self.slide_path)
        self.slide = self.read(slide_path, engine)
        self.dimensions = None
        self.level_dimensions = None
        self.level_downsamples = None
        self.properties = None
        self.offset = (0, 0)
        self.set_slide_attributes()
        self.ROI = None
        self.ROI_width = None
        self.ROI_height = None
        self.tissue_contours = None
        self.coords = None
        self.coords_attrs = None

    @staticmethod
    def retrieve_slide_name_ext(slide_path: str) -> tuple[str, str]:
        """Retrieves slide name and slide extension from slide path.

        Args:
            slide_path: The path to the slide.

        Returns:
            A tuple (slide name, slide ext).
        """
        slide_ext = re.search(r"(?<=\.)\w+$", slide_path).group(0)
        slide_name = re.search(r"[^/]+(?=\.\w+$)", slide_path).group(0)
        return slide_name, slide_ext

    @staticmethod
    def read(slide_path: str, engine: str) -> openslide.OpenSlide | tiffslide.TiffSlide:
        """Read a slide with a given engine.

        Args:
            slide_path: The path to the slide.
            engine: The backend library to use for reading the slide
                (currently only openslide and tiffslide are supported).

        Returns:
            A slide object.
        """
        if engine == "openslide":
            slide = openslide.OpenSlide(slide_path)
        elif engine == "tiffslide":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                slide = tiffslide.TiffSlide(slide_path)
        else:
            raise NotImplementedError(f"engine {engine} not supported")
        return slide

    @staticmethod
    def scale_contours(
        contours: list[np.ndarray],
        scale: int,
    ) -> list[np.ndarray]:
        """Scale the contours by a given factor.

        Args:
            contours: The contours to scale.
            scale: The scale factor to apply.

        Returns:
             The input list with scaled contours.
        """
        scaled_contours = [np.array(cont * scale, dtype="int") for cont in contours]
        return scaled_contours

    @staticmethod
    def worker_init(slide_path: str, engine: str) -> None:
        """Initialize the worker process with a wsi object.

        Args:
            slide_path: The path to the slide.
            engine: The backend library to use for reading the slide (currently only openslide
                and tiffslide are supported)
        """
        global wsi
        wsi = WSI.read(slide_path, engine)

    @staticmethod
    def is_black_white(
        patch: Image.Image,
        rgb_threshs: tuple[int, int] = (2, 220),
        percentages: tuple[float, float] = (0.6, 0.9),
    ) -> bool:
        """Check if a patch is black or white.

        Args:
            patch: The input patch.
            rgb_threshs: The tuple of thresholds for the RGB channels (black threshold, white threshold).
            percentages: The tuple of percentages of pixels below/above the thresholds to consider the patch
                as black/white.

        Returns:
            True if the patch is black or white, False otherwise.
        """
        return isBlackPatch(
            patch, rgb_thresh=rgb_threshs[0], percentage=percentages[0]
        ) or isWhitePatch(patch, rgb_thresh=rgb_threshs[1], percentage=percentages[1])

    @staticmethod
    def process_coord_candidate(
        coord: tuple[int, int],
        cont_check_fn: IsInContour | None,
        patch_level: int,
        patch_size: int,
        rgb_threshs: tuple[int, int] = (2, 220),
        percentages: tuple[float, float] = (0.6, 0.9),
    ) -> tuple[int, int] | None:
        """Determine if a candidate coordinate is valid based on a contour checking
        function and/or black/white thresholding.

        Args:
            coord: The candidate coordinate.
            cont_check_fn: The contour checking function.
            patch_level: The level at which the patch should be extracted.
            patch_size: The size of the patch (assumed to be square).
            rgb_threshs: The tuple of thresholds for the RGB channels (black threshold, white threshold).
            percentages: The tuple of percentages of pixels below/above the thresholds to consider the patch as
                black/white.

        Returns:
            The coordinate if it is valid, None otherwise.
        """
        if cont_check_fn is None or cont_check_fn(coord):
            patch = wsi.read_region(
                tuple(coord), patch_level, (patch_size, patch_size)
            ).convert("RGB")
            if not WSI.is_black_white(patch, rgb_threshs, percentages):
                return coord
            else:
                return None
        else:
            return None

    def convert_micrometer_to_pixel(
        self,
        value: float,
        level: int,
        axis: str = 'x',
    ) -> int:
        """Convert a value from micrometer to pixel.

        Args:
            value: The value to convert (in micrometer).
            level: The level at which the conversion should be performed.
            axis: The axis to use for getting the conversion factor (x or y).

        Returns:
            The input value in pixel.
        """
        return (
            int(value / float(self.properties[f"{self.engine}.mpp-{axis}"]))
            // int(self.level_downsamples[level])
        )

    def convert_pixel_to_micrometer(
        self,
        value: float,
        level: int,
        axis: str = 'x',
    ) -> float:
        """Convert a value from pixel to micrometer.

        Args:
            value: The value to convert (in pixel).
            level: The level at which the conversion should be performed.
            axis: The axis to use for getting the conversion factor (x or y).

        Returns:
            The input value in micrometer.
        """
        return (
            value
            * float(self.properties[f"{self.engine}.mpp-{axis}"])
            * self.level_downsamples[level]
        )

    def convert_units(
        self,
        value: float,
        level: int,
        from_unit: str,
        to_unit: str = "px",
        axis: str = "x",
    ) -> int:
        """Convert a value from one unit to another.

        Args:
            value: The value to convert.
            level: The level at which the conversion should be performed.
            from_unit: The unit to convert from (px or micro).
            to_unit: The unit to convert to (px or micro).
            axis: The axis to use for getting the conversion factor (x or y).

        Returns:
            The input value converted in the desired unit.
        """
        if from_unit == "micro" and to_unit == "px":
            value = self.convert_micrometer_to_pixel(value, level, axis)
        elif from_unit == "px" and to_unit == "micro":
            value = self.convert_pixel_to_micrometer(value, level, axis)
        elif from_unit == to_unit:
            pass
        else:
            raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported")
        return value

    def save_tissue_contours(
        self,
        save_dir: str,
        selected_idx: np.ndarray | None = None,
        file_format: str = "pickle",
        merge: bool = False,
        label: str | None = None,
        color: tuple[int, int, int] = (255, 0, 0),
        append_to_existing_file: bool = False,
    ) -> None:
        """Save the tissue contours in a pickle or geojson file.

        Args:
            save_dir: The path to the directory where the contours will be saved.
            selected_idx: An array of indices of the contours to save
                (if set to None, all the contours will be saved).
            file_format: The file format for saving the contours
                (pickle for python processing, geojson for QuPath processing).
            merge: Set to True to merge the contours into a single polygon (for geojson format only).
            label: An optional label to assign to the tissue contours (for geojson format only).
            color: An optional color to assign to the tissue contours (for geojson format only).
            append_to_existing_file: Set to True to append the contours to an existing geojson file.
        """
        assert self.tissue_contours is not None, (
            "No tissue contours found for the slide, "
            "please run the detect_tissue method first"
        )
        if selected_idx is not None:
            tissue_contours = [self.tissue_contours[idx] for idx in selected_idx]
        else:
            tissue_contours = self.tissue_contours
        if not os.path.isdir(save_dir):
            log.warning(f"Folder {save_dir} does not exist, creating new folder...")
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        if file_format == "pickle":
            file_path = os.path.join(save_dir, f"{self.slide_name}.pkl")
            log.info(
                f"Saving tissue contours for slide {self.slide_name} at {file_path} with pickle."
            )
            save_obj_with_pickle(tissue_contours, file_path)
        elif file_format == "geojson":
            file_path = os.path.join(save_dir, f"{self.slide_name}.geojson")
            log.info(
                f"Saving {selected_idx} tissue contours for slide {self.slide_name} at {file_path} with geojson."
            )
            polygons = contoursToPolygons(tissue_contours, merge)
            export_polygons_to_qupath(
                polygons,
                file_path,
                "annotation",
                offset=self.offset,
                label=label,
                color=color,
                append_to_existing_file=append_to_existing_file,
            )
        else:
            raise ValueError(f"format {file_format} not supported")

    def load_tissue_contours(self, file_path: str) -> None:
        """Load the tissue contours from a pickle file.

        Args:
            file_path: The path to the pickle file containing the tissue contours.
        """
        log.info(f"Loading tissue contours for slide {self.slide_name} from {file_path}.")
        self.tissue_contours = load_obj_with_pickle(file_path)

    def save_patches(
        self,
        save_dir: str,
        file_format: str = "h5",
        selected_idx: np.ndarray | None = None,
        merge: bool = False,
        label: str | None = None,
        color: tuple[int, int, int] = (255, 0, 0),
        append_to_existing_file: bool = False,
    ) -> None:
        """Save the patches in a hdf5 or geojson file.

        Args:
            save_dir: The path to the directory where the patches will be saved.
            file_format: The format for the saving (h5 for python processing, geojson for QuPath processing).
            selected_idx: An array of indices of the patches to save (if set to None, all the patches will be saved).
            merge: Set to True to merge the patches into a single polygon (for geojson format only).
            label: An optional label to assign to the patches (for geojson format only).
            color: An optional color to assign to the patches (for geojson format only).
            append_to_existing_file: Set to True to append the patches to an existing geojson file.
        """
        if selected_idx is not None:
            coords = self.coords[selected_idx]
        else:
            coords = self.coords
        if not os.path.isdir(save_dir):
            log.warning(f"Folder {save_dir} does not exist, creating new folder...")
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        if file_format == "h5":
            asset_dict = {"coords": self.coords}
            attr_dict = {"coords": self.coords_attrs}
            file_path = os.path.join(save_dir, f"{self.slide_name}.h5")
            log.info(
                f"Saving patches for slide {self.slide_name} at {file_path} with hdf5."
            )
            save_patches_with_hdf5(file_path, asset_dict, attr_dict)
        elif file_format == "geojson":
            file_path = os.path.join(save_dir, f"{self.slide_name}.geojson")
            patch_downsample = int(
                self.level_downsamples[self.coords_attrs["patch_level"]]
            )
            polygons = patchesToPolygons(
                coords, self.coords_attrs["patch_size"], patch_downsample, merge
            )
            log.info(
                f"Saving {len(coords)} patches for slide {self.slide_name} at {file_path} with geojson."
            )
            export_polygons_to_qupath(
                polygons,
                file_path,
                "annotation",
                offset=self.offset,
                label=label,
                color=color,
                append_to_existing_file=append_to_existing_file,
            )
        elif file_format == "jpg" or file_format == "png":
            log.info(
                f"Saving {len(coords)} patches for slide {self.slide_name} at {save_dir} with {file_format}."
            )
            for coord in coords:
                patch = self.read_region(
                    coord,
                    self.coords_attrs["patch_level"],
                    (self.coords_attrs["patch_size"], self.coords_attrs["patch_size"]),
                ).convert("RGB")
                patch.save(
                    os.path.join(
                        save_dir, f"{self.slide_name}_{coord[0]}_{coord[1]}.{file_format}"
                    )
                )
        else:
            raise ValueError(f"format {file_format} not supported")

    def load_patches(self, file_path: str) -> None:
        """Load the patches from a hdf5 file.

        Args:
            file_path: The path to the hdf5 file containing the patches.
        """
        log.info(f"Loading patches for slide {self.slide_name} from {file_path}.")
        self.coords, self.coords_attrs = read_h5_file(file_path, "coords")

    def set_slide_attributes(self):
        """Set the slide attributes."""
        if self.engine == "openslide":
            self.dimensions = self.slide.dimensions
            self.level_dimensions = self.slide.level_dimensions
            self.level_downsamples = self.slide.level_downsamples
            self.properties = self.slide.properties
        elif self.engine == "tiffslide":
            self.dimensions = self.slide.dimensions
            self.level_dimensions = self.slide.level_dimensions
            self.level_downsamples = self.slide.level_downsamples
            self.properties = self.slide.properties
        else:
            raise NotImplementedError(f"engine {self.engine} not supported")
        if (
            f"{self.engine}.bounds-x" in self.properties.keys()
            and self.properties[f"{self.engine}.bounds-x"] is not None
        ):
            self.offset = (
                -int(self.properties[f"{self.engine}.bounds-x"]),
                -int(self.properties[f"{self.engine}.bounds-y"]),
            )

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> Image.Image:
        """Read a region from the slide for a given level and size.

        Args:
            location: The coordinates of the top left corner of the region (in pixels).
            level: The level at which the region should be read.
            size: The size of the region (in pixels).

        Returns:
            The desired region of the slide as a PIL image.
        """
        return self.slide.read_region(location, level, size).convert("RGB")

    def create_thumbnail(
        self,
        level: int,
        crop_roi: bool = False,
    ) -> Image.Image:
        """Create a thumbnail of the slide at a given level.

        Args:
            level: The level at which the thumbnail should be created.
            crop_roi: A boolean to crop the thumbnail to the region of interest defined for the slide
                (requires a ROI to be set for the slide beforehand with the
                [set_roi][prismtoolbox.wsicore.WSI.set_roi] method)

        Returns:
            A thumbnail of the slide as a PIL image.
        """
        assert self.ROI is not None if crop_roi else True, (
            "No ROI provided, while crop_roi is set to True. Please set a ROI for "
            "the slide or set crop_roi to False."
        )
        thumb = self.read_region((0, 0), level, self.level_dimensions[level]).convert(
            "RGB"
        )
        if crop_roi:
            log.info(f"Creating thumbnail with ROI {self.ROI}.")
            coords_roi = (self.ROI / self.level_downsamples[level]).astype(int)
            thumb = thumb.crop(coords_roi)
        return thumb

    def set_roi(
        self,
        roi: tuple[int, int, int, int] | None = None,
        rois_df_path: str | None = None,
    ) -> None:
        """Set the region of interest for the slide. Can be set manually or by selecting
        a region on a thumbnail. The ROI is stored as a tuple in the ROI attribute.

        Args:
            roi: Set the region of interest manually as a tuple (x1, y1, x2, y2).
            rois_df_path: The path to dataframe containing the ROIs with a slide_id column identifying the slide.

        Returns:
            The region of interest set for the slide as a tuple (x1, y1, x2, y2).
        """
        if roi is not None:
            ROI = np.array(roi).astype(int)
        elif rois_df_path is not None:
            rois_df = pd.read_csv(rois_df_path)
            ROI = rois_df[rois_df.slide_id == self.slide_name].values[0].astype(int)
        else:
            log.info("No ROI provided, prompting user to select one.")
            level = input(
                f"No ROI was provided, please select a level at which the ROI should be created (max level: {len(self.level_dimensions)-1}): "
            )
            if not level:
                log.warning("No level provided, setting the ROI at the highest level.")
                level = len(self.level_downsamples) - 1
            else:
                level = int(level)
            img = np.array(self.create_thumbnail(level))
            ROI = select_roi_on_thumbnail(img, int(self.level_downsamples[level]))
            ROI = (ROI * self.level_downsamples[level]).astype(int)
        self.ROI = ROI
        self.ROI_width = ROI[2] - ROI[0]
        self.ROI_height = ROI[3] - ROI[1]
        print(f"ROI for slide {self.slide_name} has been set to {self.ROI}.")
        return ROI

    def detect_tissue(
        self,
        seg_level: int,
        window_avg: int,
        window_eng: int,
        thresh: int,
        inside_roi: bool = False,
        inv_thresh: bool = False,
        area_min: float = 2e5,
        start: tuple[int, int] = (0, 0),
    ) -> None:
        """Segment the tissue on the slide based on a threshold on the Law's texture
        energy spot map and floodfill algorithm to fill the holes in the mask. The
        tissue contours are stored in the tissue_contours attribute.

        Args:
            seg_level: The level at which the segmentation should be performed.
            window_avg: The size of the window for local averaging.
            window_eng: The size of the window for Law's texture energy computation.
            thresh: The threshold for binarization on the Law's texture energy spot map.
            inside_roi: Set to True to identify the tissue only within a ROI (requires a ROI to be set for the slide
                beforehand with the [set_roi][prismtoolbox.wsicore.WSI.set_roi] method).
            inv_thresh: Set to True to invert the thresholding.
            area_min: The minimum area for a contour to be kept.
            start: The starting point for the floodfill algorithm (should be left at (0, 0) in most cases).
        """
        final_contours = []
        img = np.array(self.create_thumbnail(seg_level, inside_roi))
        img_avg = local_average(np.asarray(img), window_avg)
        law_feats = compute_law_feats(img_avg, window_eng)[0]
        filterred_img = apply_bilateral_filter(np.clip(law_feats, 0, 255).astype("uint8"))
        threshed_img = apply_binary_thresh(filterred_img, thresh, inv_thresh)
        flooded_img = floodfill_img(np.pad(threshed_img, 1), start)
        contours = contour_mask(flooded_img)
        for contour in contours:
            c = contour.copy()
            area = cv2.contourArea(c)
            if area > area_min:
                final_contours.append(contour)
        if len(final_contours) == 0:
            self.tissue_contours = []
            print(f"No contours found for slide {self.slide_name}.")
            return
        else:
            scale = self.level_downsamples[seg_level]
            offset = np.array(self.ROI[:2]) if self.ROI is not None else np.array([0, 0])
            final_contours = self.scale_contours(final_contours, scale)
            final_contours = [cont + offset for cont in final_contours]
            # Sanity check to ensure that the contours are all within a ROI is provided
            if self.ROI is not None:
                assert all(
                    [
                        np.all(cont >= self.ROI[:2]) and np.all(cont <= self.ROI[2:])
                        for cont in final_contours
                    ]
                )
            self.tissue_contours = final_contours
            print(
                f"Identified {len(final_contours)} contours for slide {self.slide_name}."
            )
            return

    def apply_pathologist_annotations(self, path: str) -> None:
        """Apply pathologist annotations to the tissue contours. Requires the tissue
        contours to be set for the slide beforehand with the
        [detect_tissue][prismtoolbox.wsicore.WSI.detect_tissue] method.

        Args:
            path: The path to the .geojson file containing the annotations extracted from QuPath.
        """
        assert (
            self.tissue_contours is not None
        ), "No tissue contours found for the slide, please run the detect_tissue method first"
        offset = (
            (
                int(self.properties[f"{self.engine}.bounds-x"]),
                int(self.properties[f"{self.engine}.bounds-y"]),
            )
            if "{self.engine}.bounds-x" in self.properties
            else (0, 0)
        )
        pathologist_annotations = read_qupath_annotations(path, offset=offset)
        polygons = contoursToPolygons(self.tissue_contours)
        intersection = intersectionPolygons(polygons, pathologist_annotations)
        self.tissue_contours = PolygonsToContours(intersection)

    def extract_patches(
        self,
        patch_size: float,
        patch_level: int,
        mode: str,
        step_size: float | None = None,
        overlap: float | None = None,
        units: tuple[str, str] = ("px", "px"),
        use_padding: bool = True,
        contours_mode: str | None = None,
        rgb_threshs: tuple[int, int] = (2, 220),
        percentages: tuple[float, float] = (0.6, 0.9),
    ) -> None:
        """Extract valid patches from the slide with different extraction modes. A patch
        is considered valid if it is not black or white and is within the region of
        interest or the tissue contours if relevant. The extracted patches are stored as
        coordinates in the coords attribute, and the attributes of the coordinates are
        stored in the coords_attrs attribute.

        Args:
            patch_size: The size of the patches to extract (assumed to be square).
            patch_level: The level at which the patches should be extracted.
            mode: The mode to use for the extraction:

                - "contours" mode extracts patches within the tissue contours (requires the tissue contours
                to be set for the slide beforehand with the [detect_tissue][prismtoolbox.wsicore.WSI.detect_tissue]
                method).
                - "roi" mode extracts patches within the region of interest (requires the ROI to be set
                for the slide beforehand with the [set_roi][prismtoolbox.wsicore.WSI.set_roi] method).
                - "all" mode extracts patches from the entire slide

            step_size: The step size for the sliding window (if set to None, the step size will be computed based on the
                overlap).
            overlap: The overlap between patches as an absolute value (must be provided if step_size is set to None).
            units: The units for the patch size and step size/overlap values (pixels: px, micrometers: micro).
            use_padding: Set to True to use padding for the extraction.
            contours_mode: The mode to use for the contour checking function (must be provided if mode is
                set to contours). See [IsInContour][prismtoolbox.wsicore.core_utils.IsInContour] for
                more details.
            rgb_threshs: The tuple of thresholds for the RGB channels (black threshold, white threshold).
            percentages: The tuple of percentages of pixels below/above the thresholds to consider the patch as
                black/white.
        """
        assert (
            step_size is not None or overlap is not None
        ), "Either step_size or overlap must be provided"
        assert all(
            [unit in ["micro", "px"] for unit in units]
        ), "Units must be either 'micro' or 'px'"

        patch_size = self.convert_units(patch_size, patch_level, units[0])
        if step_size is None:
            step_size = patch_size - self.convert_units(overlap, patch_level, units[1])
        else:
            step_size = self.convert_units(step_size, patch_level, units[1])

        log.info(
            f"Extracting patches of size {patch_size} at level {patch_level} with step size {step_size}."
        )

        if mode == "contours":
            log.info("Extracting patches with 'contours' mode.")
            assert self.tissue_contours is not None
            assert len(self.tissue_contours) > 0
            assert contours_mode is not None
            valid_coords = []
            for cont in self.tissue_contours:
                roi_dim = cv2.boundingRect(cont)
                log.info(f"Processing ROI of dimensions: {roi_dim}")
                valid_coords.extend(
                    self.extract_patches_roi(
                        roi_dim,
                        patch_level,
                        patch_size,
                        step_size,
                        use_padding,
                        cont,
                        contours_mode,
                        rgb_threshs,
                        percentages,
                    )
                )
            valid_coords = np.array(valid_coords)
        elif mode == "roi":
            log.info("Extracting patches with 'roi' mode.")
            assert self.ROI is not None
            roi_dim = self.ROI[0], self.ROI[1], self.ROI_width, self.ROI_height
            log.info("Processing ROI of dimensions:", roi_dim)
            valid_coords = self.extract_patches_roi(
                roi_dim,
                patch_level,
                patch_size,
                step_size,
                use_padding,
                rgb_threshs=rgb_threshs,
                percentages=percentages,
            )
        elif mode == "all":
            roi_dim = 0, 0, self.level_dimensions[0][0], self.level_dimensions[0][1]
            log.info("Processing ROI of dimensions:", roi_dim)
            valid_coords = self.extract_patches_roi(
                roi_dim,
                patch_level,
                patch_size,
                step_size,
                use_padding,
                rgb_threshs=rgb_threshs,
                percentages=percentages,
            )
        else:
            raise ValueError(f"Mode {mode} not supported")

        attr = {
            "patch_size": patch_size,
            "patch_level": patch_level,
            "downsample": self.level_downsamples[patch_level],
            "downsampled_level_dim": tuple(np.array(self.level_dimensions[patch_level])),
            "level_dim": self.level_dimensions[patch_level],
            "name": self.slide_name,
        }

        if len(valid_coords) == 0:
            log.warning(f"No valid coordinates found for slide {self.slide_name}.")
        else:
            print(
                f"Identified a total of {len(valid_coords)}  valid coordinates in the slide {self.slide_name}."
            )
            self.coords = valid_coords
            self.coords_attrs = attr

    def extract_patches_roi(
        self,
        roi_dim: tuple[int, int, int, int],
        patch_level: int,
        patch_size: int,
        step_size: int,
        use_padding: bool = True,
        contour: np.ndarray | None = None,
        contours_mode: str | None = None,
        rgb_threshs: tuple[int, int] = (2, 220),
        percentages: tuple[float, float] = (0.6, 0.9),
    ) -> np.ndarray:
        """Extract valid patches from a region of interest, i.e if the patch is not
        black or white and is within the region of interest/contours if relevant).

        Args:
            roi_dim: The top-left corner coordinates and dimensions of the region of interest.
            patch_level: The level at which the patches should be extracted.
            patch_size: The size of the patches to extract (assumed to be square).
            step_size: The step size to use for the sliding window.
            use_padding: Set to True to use padding for the extraction.
            contour: The tissue contour to use for the extraction. If set to None, will not check if patches are within
                a contour.
            contours_mode: The mode for the contour checking function.
                See [IsInContour][prismtoolbox.wsicore.core_utils.IsInContour] for more details. Must be provided
                if mode is set to contours. Otherwise, will not check if patches are within the contours
            rgb_threshs: The tuple of thresholds for the RGB channels (black threshold, white threshold).
            percentages: The tuple of percentages of pixels below/above the thresholds to consider the patch as
                black/white.

        Returns:
            An array of valid coordinates for the patches (i.e. coordinates of the top-left corner of the patches).
        """
        assert contours_mode is not None if contour is not None else True, (
            "A contour mode must be provided if patch "
            "extraction mode is set to contours"
        )
        start_x, start_y, w, h = roi_dim

        patch_downsample = int(self.level_downsamples[patch_level])
        ref_patch_size = patch_size * patch_downsample

        img_w, img_h = self.level_dimensions[0]

        if use_padding:
            stop_y = start_y + h
            stop_x = start_x + w
        else:
            stop_y = min(start_y + h, img_h - ref_patch_size + 1)
            stop_x = min(start_x + w, img_w - ref_patch_size + 1)

        step_size = step_size * patch_downsample

        x_range = np.arange(start_x, stop_x, step=step_size)
        y_range = np.arange(start_y, stop_y, step=step_size)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing="ij")
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

        if contour is not None:
            cont_check_fn = IsInContour(
                contour, patch_size=ref_patch_size, center_shift=0.5, mode=contours_mode
            )
            log.info(
                f"Extracting patches with contour checking function mode {contours_mode}"
            )
        else:
            cont_check_fn = None

        num_workers = mp.cpu_count()
        pool = mp.Pool(
            num_workers,
            initializer=WSI.worker_init,
            initargs=(
                self.slide_path,
                self.engine,
            ),
        )
        iterable = [
            (coord, cont_check_fn, patch_level, patch_size, rgb_threshs, percentages)
            for coord in coord_candidates
        ]
        valid_coords = pool.starmap(WSI.process_coord_candidate, iterable)
        pool.close()

        valid_coords = np.array([coord for coord in valid_coords if coord is not None])

        log.info(
            f"Identified {len(valid_coords)}  valid coordinates in the ROI {roi_dim}."
        )

        return valid_coords

    def visualize(
        self,
        vis_level: int,
        crop_roi: bool = False,
        contours_color: tuple[int, int, int] = (255, 0, 0),
        line_thickness: int = 500,
        max_size: int | None = None,
        number_contours: bool = False,
        view_slide_only: bool = False,
    ) -> Image.Image:
        """Visualize the slide with or without the contours of the tissue.

        Args:
            vis_level: The level at which the visualization should be performed.
            crop_roi: Set to True to crop the visualization to the region of interest (requires a ROI to be set for the
                slide beforehand with the [set_roi][prismtoolbox.wsicore.WSI.set_roi] method).
            contours_color: The color to use for the contours.
            line_thickness: The thickness to use for the contours
            max_size: The maximum size for the visualization for the width or height of the image.
            number_contours: Set to True to number the contours.
            view_slide_only: Set to True to visualize the slide only (without the contours).

        Returns:
            A PIL image of the visualization.
        """
        scale = 1 / self.level_downsamples[vis_level]

        img = np.array(self.create_thumbnail(vis_level, crop_roi))

        line_thickness = int(line_thickness * scale)

        if not view_slide_only:
            assert len(self.tissue_contours) > 0, (
                "No tissue contours found for the slide, please run the detect_tissue method first"
            )
            offset = self.ROI[:2] if crop_roi else np.array([0, 0])
            contours = [cont - offset for cont in self.tissue_contours]
            contours = self.scale_contours(contours, scale)
            line_thickness = int(line_thickness * scale)
            if len(contours) > 0:
                if not number_contours:
                    cv2.drawContours(
                        img,
                        contours,
                        -1,
                        contours_color,
                        line_thickness,
                        lineType=cv2.LINE_8,
                    )
                else:  # add numbering to each contour
                    for idx, cont in enumerate(contours):
                        M = cv2.moments(cont)
                        cX = int(M["m10"] / (M["m00"] + 1e-9))
                        cY = int(M["m01"] / (M["m00"] + 1e-9))
                        # draw the contour and put text next to center
                        cv2.drawContours(
                            img,
                            [cont],
                            -1,
                            contours_color,
                            line_thickness,
                            lineType=cv2.LINE_8,
                        )
                        cv2.putText(
                            img,
                            "{}".format(idx),
                            (cX, cY),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (255, 0, 0),
                            10,
                        )

        img = Image.fromarray(img)

        w, h = img.size

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size / w if w > h else max_size / h
            img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

        return img

    def stitch(
        self,
        vis_level: int,
        selected_idx: np.ndarray = None,
        colors: np.ndarray = None,
        alpha: float = 0.6,
        black_white: bool = False,
        draw_grid: bool = False,
        crop_roi: bool = False,
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> Image.Image:
        """Stitch the patches extracted on an image. The patches can be masked and
        colored depending on the mask and colors provided. Requires the coordinates of
        the patches to be set for the slide beforehand with the
        [extract_patches][prismtoolbox.wsicore.WSI.extract_patches] method.

        Args:
            vis_level: The level at which the patches should be visualized.
            selected_idx: An array of indices of the patches to visualize (if set to None, all the patches will be
                visualized).
            colors: An array of RGB colors to apply to the patches (if set to None, the patches will be visualized as
                they are).
            alpha: Set the transparency of the colors to apply to the patches.
            black_white: Set to True to visualize a binary mask of the patches extracted.
            draw_grid: Set to True to draw a grid on the stitched patches.
            crop_roi: Set to True to crop the visualization to the region of interest (requires a ROI to be set for the
                slide beforehand with the [set_roi][prismtoolbox.wsicore.WSI.set_roi] method).
            background_color: The color of the background.

        Returns:
            A PIL image of the stitched patches.
        """
        assert self.ROI is not None if crop_roi else True, (
            "no ROI provided, while crop_roi is set to True,"
            " please set a ROI for the slide or set crop_roi to False."
        )
        assert self.coords is not None, (
            "no coordinates provided for the patches to visualize, please run the "
            "extract_patches method first or load the coordinates from a file"
        )
        if crop_roi:
            w, h = int(np.ceil(self.ROI_width / self.level_downsamples[vis_level])), int(
                np.ceil(self.ROI_height / self.level_downsamples[vis_level])
            )
        else:
            w, h = self.level_dimensions[vis_level]
        patch_size = self.coords_attrs["patch_size"]
        patch_level = self.coords_attrs["patch_level"]
        patch_size = int(patch_size * self.level_downsamples[patch_level])
        canvas = init_image(w, h, mask=black_white, color_bakground=background_color)
        downsample_vis = self.level_downsamples[vis_level]
        idxs = np.arange(len(self.coords))
        if selected_idx is not None:
            idxs = idxs[selected_idx]
        patch_size = np.ceil(patch_size / downsample_vis).astype(int)
        log.info(
            f"Stitching {len(idxs)} patches at level {vis_level} with patch size {patch_size},"
            f"with colors {colors is not None}."
        )
        offset = self.ROI[:2] if crop_roi else np.array([0, 0])
        for idx in idxs:
            coord = self.coords[idx]
            coord_downsampled = np.ceil(np.abs(coord - offset) / downsample_vis).astype(int)
            patch_size_coord = (
                min(max(w - coord_downsampled[0], 0), patch_size),
                min(max(h - coord_downsampled[1], 0), patch_size),
            )
            if any(val == 0 for val in patch_size_coord):
                continue
            if black_white:
                patch = np.ones(patch_size_coord, dtype="uint8")
                colors = None
            else:
                patch = np.array(
                    self.slide.read_region(
                        tuple(coord), vis_level, patch_size_coord
                    ).convert("RGB")
                )
            if colors is not None:
                assert len(colors) == len(idxs), (
                    "the number of colors provided must match "
                    "the number of selected coordinates"
                )
                color = colors[idx]
                color_patch = (
                    np.ones((patch_size_coord[1], patch_size_coord[0], 3)) * color
                ).astype("uint8")
                canvas[
                    coord_downsampled[1] : coord_downsampled[1] + patch_size_coord[1],
                    coord_downsampled[0] : coord_downsampled[0] + patch_size_coord[0],
                    :,
                ] = cv2.addWeighted(color_patch, alpha, patch, 1 - alpha, 0, patch)
            else:
                canvas[
                    coord_downsampled[1] : coord_downsampled[1] + patch_size_coord[1],
                    coord_downsampled[0] : coord_downsampled[0] + patch_size_coord[0],
                    :,
                ] = patch
            if draw_grid:
                cv2.rectangle(
                    canvas,
                    tuple(np.maximum([0, 0], coord_downsampled - 1)),
                    tuple(coord_downsampled + patch_size_coord),
                    (0, 0, 0, 255),
                    thickness=2,
                )
        img = Image.fromarray(canvas)
        return img

    def __repr__(self):
        return f"WSI({self.slide_path}, {self.engine}) with level {self.level_dimensions}"
