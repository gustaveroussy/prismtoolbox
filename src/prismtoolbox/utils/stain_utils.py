import numpy as np
import skimage.color as skcolor

IHC_custom = [
    [1.06640743, 1.081867, 0.85172556],
    [0.71290748, 0.95542728, 1.33166524],
    [-0.62692285, 0.81289619, -0.24760368],
]


def retrieve_conv_matrix(conv_matrix_name="HED"):
    if conv_matrix_name == "HED":
        conv_matrix = skcolor.hed_from_rgb
    elif conv_matrix_name == "HD":
        conv_matrix = skcolor.hdx_from_rgb
    elif conv_matrix_name == "HD_custom": # to fix
        conv_matrix = np.linalg.inv(IHC_custom)
    else:
        raise ValueError("conv_matrix_name must be 'HED', 'HD' or 'HD_custom'")
    return conv_matrix

def deconvolve_img(img, conv_matrix_name="HED"):
    conv_matrix = retrieve_conv_matrix(conv_matrix_name)
    stains = skcolor.separate_stains(img, conv_matrix)
    null = np.zeros_like(stains[:, :, 0])
    img_a = skcolor.combine_stains(
        np.stack((stains[:, :, 0], null, null), axis=-1),
        np.linalg.inv(conv_matrix),
    )
    img_b = skcolor.combine_stains(
        np.stack((null, stains[:, :, 1], null), axis=-1),
        np.linalg.inv(conv_matrix),
    )
    img_c = skcolor.combine_stains(
        np.stack((null, null, stains[:, :, 2]), axis=-1),
        np.linalg.inv(conv_matrix),
    )
    return (
        (img_a * 255).astype("uint8"),
        (img_b * 255).astype("uint8"),
        (img_c * 255).astype("uint8"),
    )
