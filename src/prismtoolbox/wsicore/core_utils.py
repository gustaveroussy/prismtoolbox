from __future__ import annotations

import h5py
import numpy as np
import cv2
from PIL import Image
from scipy.signal import oaconvolve
from itertools import product


def select_roi_on_thumbnail(img: np.ndarray, scale_factor: int) -> np.ndarray:
    """Select a region of interest on the thumbnail of the slide using an interactive
    window.

    Args:
        img: A thumbnail of the slide.
        scale_factor: The scale factor to apply to convert the coordinates to the original slide dimensions.

    Returns:
        The coordinates of the selected region of interest.
    """
    drawing = False
    roi = [0, 0, 0, 0]

    def _draw_rectangle(event, x, y, flags, param):
        nonlocal roi, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            roi[0], roi[1] = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                roi[2], roi[3] = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi[2], roi[3] = x, y

    window_name = "Select ROI (beginning by the top left corner) - press ESC to exit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 800)
    cv2.setMouseCallback(window_name, _draw_rectangle)

    while True:
        temp_img = img.copy()
        if roi[2] and roi[3]:
            cv2.rectangle(temp_img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 5)
            width = (roi[2] - roi[0]) * scale_factor
            height = (roi[3] - roi[1]) * scale_factor
            cv2.putText(
                temp_img,
                f"Width: {width} - Height: {height}",
                (roi[0], roi[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        cv2.imshow(window_name, temp_img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    return np.array(roi)


def local_average(img: np.ndarray, window_size: int) -> np.ndarray:
    """Perform local averaging on the image for a given window size.

    Args:
        img: An input image.
        window_size: The window size to use for local averaging.

    Returns:
        Grayscale image minus the local average on a window of size window_size of the input image.
    """
    window = np.ones((window_size, window_size)) / (window_size**2)
    img_grayscaled = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_grayscaled - oaconvolve(img_grayscaled, window, mode="same")


def compute_law_feats(
    img: np.ndarray,
    window_size: int,
    only_s5: bool = True
) -> list[np.ndarray]:
    """Compute the Law's texture energy for a given grayscale image and window size.

    Args:
      img: An input grayscale image.
      window_size: The window size to use for Law's texture energy computation.
      only_s5: Set to True to only compute the Law's texture energy for the S5S5 filter.

    Returns:
        A list of Law's texture energy map for each filter extracted from the input image.
    """
    L5 = np.array([1, 4, 6, 4, 1])
    E5 = np.array([-1, -2, 0, 2, 1])
    S5 = np.array([-1, 0, 2, 0, -1])
    R5 = np.array([1, -4, 6, -4, 1])

    vectors = [L5, E5, S5, R5]
    vectors_name = ["L5", "E5", "S5", "R5"]
    if only_s5:
        vectors = [vectors[2]]
        vectors_name = [vectors_name[2]]

    filters = [
        np.expand_dims(vectors[i], -1).dot(np.expand_dims(vectors[j], -1).T)
        for i, j in product(range(len(vectors)), range(len(vectors)))
    ]
    filters_name = np.array(
        [
            vectors_name[i] + vectors_name[j]
            for i, j in product(range(len(vectors)), range(len(vectors)))
        ]
    )

    imgs_filtered = []
    for filt in filters:
        imgs_filtered.append(oaconvolve(img, filt, mode="same"))

    window = np.ones((window_size, window_size))
    imgs_energy = []
    for img_filtered in imgs_filtered:
        imgs_energy.append(oaconvolve(np.abs(img_filtered), window, mode="same"))

    def _get_img_energy(name):
        return imgs_energy[np.where(filters_name == name)[0].item()]

    if only_s5:
        imgs_feats = [_get_img_energy("S5S5")]
    else:
        imgs_feats = [
            np.mean(np.array([_get_img_energy("L5E5"), _get_img_energy("E5L5")]), axis=0),
            np.mean(np.array([_get_img_energy("L5R5"), _get_img_energy("R5L5")]), axis=0),
            np.mean(np.array([_get_img_energy("E5S5"), _get_img_energy("S5E5")]), axis=0),
            _get_img_energy("S5S5"),
            _get_img_energy("R5R5"),
            np.mean(np.array([_get_img_energy("L5S5"), _get_img_energy("S5L5")]), axis=0),
            _get_img_energy("E5E5"),
            np.mean(np.array([_get_img_energy("E5R5"), _get_img_energy("R5E5")]), axis=0),
            np.mean(np.array([_get_img_energy("S5R5"), _get_img_energy("R5S5")]), axis=0),
        ]

    return imgs_feats


def apply_bilateral_filter(img: np.ndarray) -> np.ndarray:
    """Apply a bilateral filter on a grayscale image.

    Args:
        img: An input grayscale image.

    Returns:
        The input image after applying a bilateral filter of parameters 3, 3*2, 3/2.
    """
    assert len(img.shape) == 2, f"Input image should be grayscale, got shape {img.shape}"
    img_filtered = cv2.bilateralFilter(img, 3, 3 * 2, 3 / 2)
    return img_filtered


def apply_binary_thresh(img: np.ndarray, thresh: int, inv: bool) -> np.ndarray:
    """Apply a binary threshold on a grayscale image.

    Args:
        img: An input grayscale image.
        thresh: The threshold value.
        inv: Set to True to invert the binary threshold.

    Returns:
        The input image after applying a binary threshold.
    """
    if inv:
        _, img_thresh = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
    else:
        _, img_thresh = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return img_thresh


def floodfill_img(img: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Perform the floodfill algorithm on a binary image.

    Args:
        img: A binary image.
        start: coordinates of the starting point for the floodfill algorithm.

    Returns:
        The input image after applying the floodfill algorithm.
    """
    img_to_floodfill = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(img_to_floodfill, mask, start, 255)
    im_floodfill_inv = cv2.bitwise_not(img_to_floodfill)
    img_out = img | im_floodfill_inv
    return img_out[1:-1, 1:-1]


def contour_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Find the contours in a binary floodfilled image with opencv findContours.

    Args:
        mask: A binary floodfilled image.

    Returns:
        A list of contours found in the input image with opencv findContours.
    """
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


class IsInContour:
    def __init__(
        self, 
        contour: np.ndarray,
        patch_size: int, 
        center_shift: int = 0.5,
        mode: str = "center"
    ):
        """The IsInContour class checks if a patch is inside a contour.

        Args:
            contour: The contour to check if the patch is inside.
            patch_size: The size of the patches that will be checked.
            center_shift: The shift of the center of the patch.
            mode: The mode to use for the contour checking. Possible values are:

                - "center" mode checks if the center of the patch is within the contour.
                - "four_pt" mode checks if one out of the four corners of the patch are within the contour.
                - "four_pt_hard" mode checks if the four corners of the patch are within the contour.
        """
        self.cont = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)
        self.mode = mode

    def __call__(self, pt: tuple[int, int]) -> int:
        """Check if a point is inside the contour.

        Args:
            pt: The top left coordinate of the patch.

        Returns:
            1 if the patch is inside the contour, 0 otherwise.
        """
        center = (int(pt[0] + self.patch_size // 2), int(pt[1] + self.patch_size // 2))
        if self.mode == "center":
            return 1 if cv2.pointPolygonTest(self.cont, center, False) >= 0 else 0
        else:
            if self.shift > 0:
                all_points = [
                    (center[0] - self.shift, center[1] - self.shift),
                    (center[0] + self.shift, center[1] + self.shift),
                    (center[0] + self.shift, center[1] - self.shift),
                    (center[0] - self.shift, center[1] + self.shift),
                ]
            else:
                all_points = [center]
            # Easy version of 4pt contour checking function - 1 of 4 points need to be in the contour for test to pass
            if self.mode == "four_pt":
                for points in all_points:
                    if cv2.pointPolygonTest(self.cont, points, False) >= 0:
                        return 1
                return 0
            # Hard version of 4pt contour checking function - all 4 points need to be in the contour for test to pass
            elif self.mode == "four_pt_hard":
                for points in all_points:
                    if cv2.pointPolygonTest(self.cont, points, False) < 0:
                        return 0
                return 1
            else:
                raise ValueError(f"mode {self.mode} not recognized")


def isBlackPatch(
    patch: Image.Image,
    rgb_thresh: int = 20,
    percentage: float = 0.05
) -> bool:
    """Check if a patch is black.

    Args:
        patch: An input patch.
        rgb_thresh: The threshold value for the RGB channels to be considered black.
        percentage: The percentage of pixels below the threshold to consider the patch as black.

    Returns:
        True if the patch is black, False otherwise.
    """
    num_pixels = patch.size[0] * patch.size[1]
    return (
        True
        if np.all(np.array(patch) < rgb_thresh, axis=2).sum() > num_pixels * percentage
        else False
    )


def isWhitePatch(
    patch: Image.Image,
    rgb_thresh: int = 220,
    percentage: float = 0.2
) -> bool:
    """Check if a patch is white.

    Args:
        patch: An input patch.
        rgb_thresh: The threshold value for the RGB channels to be considered white.
        percentage: The percentage of pixels above the threshold to consider the patch as white.

    Returns:
        True if the patch is white, False otherwise.
    """
    num_pixels = patch.size[0] * patch.size[1]
    return (
        True
        if np.all(np.array(patch) > rgb_thresh, axis=2).sum() > num_pixels * percentage
        else False
    )


def save_patches_with_hdf5(
    output_path: str,
    asset_dict: dict[str, np.ndarray],
    attr_dict: dict[str, dict[str, str | int | tuple[int, int]]] = None,
) -> None:
    """Save patches to an HDF5 file.

    Args:
      output_path: The path to the output HDF5 file.
      asset_dict: A dictionary of patches to save (key: name of the dataset, value: array of patches coordinates).
      attr_dict: A dictionary of attributes to save (key: name of the dataset, value: dictionary of attributes to
    set to the HDF5 dataset).
    """
    file = h5py.File(output_path, "w")
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(
                key,
                shape=data_shape,
                maxshape=maxshape,
                chunks=chunk_shape,
                dtype=data_type,
            )
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
