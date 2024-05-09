import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from PIL import Image, ImageOps
from typing import Tuple, Optional


def init_image(
    w: int,
    h: int,
    color_bakground: Optional[Tuple[int, int, int]] = (255, 255, 255),
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


def bbox_from_coords(
    coords: np.ndarray, patch_size: Optional[int] = 0
) -> Tuple[int, int, int, int]:
    """
    Compute the bounding box from a set of coordinates
    :param coords: coordinates of the patches
    :param patch_size: size of the patches
    :return: bounding box of the patches
    """
    x_min, y_min = coords.min(axis=0).astype(int)
    x_max, y_max = coords.max(axis=0).astype(int)
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


def plot_scatter(tx, ty, labels, cmap):
    colors = get_colors_from_cmap(cmap, len(np.unique(labels)), scale=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for k, label in enumerate(np.unique(labels)):
        # find the samples of the current class in the data
        indices = [i for i in np.arange(len(labels)) if labels[i] == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=[colors[k]], label=label)

    # build a legend using the labels we set previously
    ax.legend(loc="best")
    plt.gca().invert_yaxis()
    # finally, show the plot
    plt.show()