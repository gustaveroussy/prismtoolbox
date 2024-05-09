import numpy as np
import skimage.color as skcolor

conv_matrix = [
    [1.06640743, 1.081867, 0.85172556],
    [0.71290748, 0.95542728, 1.33166524],
    [-0.62692285, 0.81289619, -0.24760368],
]


def deconvolve_stain(img, conv_matrix="HE"):
    img = np.array(img)
    if conv_matrix == "HE":
        conv_matrix = skcolor.hed_from_rgb
    elif conv_matrix == "IHC":
        conv_matrix = skcolor.hdx_from_rgb
    elif conv_matrix == "IHC_custom": # to fix
        conv_matrix = np.lubgalg.inv(conv_matrix)
    else:
        raise ValueError("conv_matrix must be 'HE', 'IHC' or 'IHC_custom'")
    deconvole_img = skcolor.separate_stains(img, conv_matrix)
    null = np.zeros_like(deconvole_img[:, :, 0])
    img_a = skcolor.combine_stains(
        np.stack((deconvole_img[:, :, 0], null, null), axis=-1),
        np.linalg.inv(conv_matrix),
    )
    img_b = skcolor.combine_stains(
        np.stack((null, deconvole_img[:, :, 1], null), axis=-1),
        np.linalg.inv(conv_matrix),
    )
    img_c = skcolor.combine_stains(
        np.stack((null, null, deconvole_img[:, :, 2]), axis=-1),
        np.linalg.inv(conv_matrix),
    )
    return (
        (img_a * 255).astype("uint8"),
        (img_b * 255).astype("uint8"),
        (img_c * 255).astype("uint8"),
    )
