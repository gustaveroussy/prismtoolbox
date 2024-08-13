from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image, ImageOps


def init_image(
    w: int,
    h: int,
    color_bakground: tuple[int, int, int] = (255, 255, 255),
    mask: bool = False,
) -> np.ndarray:
    """
    Create an image with the dimension (w,h) and the color of the background set to color_bakground
    :param w: width of image
    :param h: height of image
    :param color_bakground: color of the background
    :param mask: set to True to create a binary mask
    :return: PIL image with the dimension (w,h), in grayscale when mask if set to True
    """
    if mask:
        return np.array(ImageOps.grayscale(Image.new(size=(w, h), mode="RGB", color=0)))
    else:
        return np.array(Image.new(size=(w, h), mode="RGB", color=color_bakground))


def bbox_from_contours(
    contours: list[np.ndarray], downsample_factor: int = 1
) -> tuple[int, int, int, int]:
    """
    Compute the bounding box from a set of contours
    :param contours: list of contours
    :return: bounding box of the contours
    """
    flatten_contours = np.concatenate(contours)
    x_min, y_min = flatten_contours.min(axis=0).squeeze().astype(int) / downsample_factor
    x_max, y_max = flatten_contours.max(axis=0).squeeze().astype(int) / downsample_factor
    return x_min, y_min, x_max, y_max


def bbox_from_coords(
    coords: np.ndarray,
    patch_size: int = 0,
    downsample_factor: int = 1,
) -> tuple[int, int, int, int]:
    """
    Compute the bounding box from a set of coordinates
    :param coords: coordinates of the patches
    :param patch_size: size of the patches (at level extraction)
    :return: bounding box of the patches
    """
    x_min, y_min = coords.min(axis=0).astype(int) / downsample_factor
    x_max, y_max = coords.max(axis=0).astype(int) / downsample_factor
    return x_min, y_min, x_max + patch_size, y_max + patch_size


def get_colors_from_cmap(cmap_name, n_colors, scale=255):
    """
    Get a list of colors from a matplotlib colormap
    :param cmap_name: name of a matplotlib colormap
    :param n_colors: number of colors
    :return: list of colors
    """
    cmap_name = plt.get_cmap(cmap_name, n_colors)
    colors_from_cmap = np.array([to_rgb(cmap_name(i)) for i in range(n_colors)])
    if scale > 1:
        return (colors_from_cmap * scale).astype(int)
    else:
        return colors_from_cmap


def plot_scatter(tx, ty, cmap=None, labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if labels is None:
        ax.scatter(tx, ty)
    else:
        if cmap is None:
            cmap = "Set1"
        colors = get_colors_from_cmap(cmap, len(np.unique(labels)), scale=1)
        for k, label in enumerate(np.unique(labels)):
            indices = [i for i in np.arange(len(labels)) if labels[i] == label]

            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            ax.scatter(current_tx, current_ty, c=[colors[k]], label=label)

        ax.legend(loc="best")
        plt.gca().invert_yaxis()

    plt.show()
