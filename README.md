# PrismToolBox

PrismToolBox is a comprehensive toolkit designed to streamline the process of working with digital pathology images. It provides a set of utilities and functionalities that simplify tasks such as reading slides, extracting patches, and embedding slides. It also provides useful functions to import/export data on [QuPath](https://qupath.github.io/).

The toolkit is still under development, and some features may not be fully implemented. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Features

- **Slide Reading**: Supports multiple backends for reading slides, including OpenSlide and TiffSlide.
- **Patch Extraction**: Extracts patches from slides at various magnification levels.
- **Slide Embedding**: Uses pre-trained models to generate embeddings from patches, facilitating downstream tasks like clustering or classification (in progress).
- **Nuclei segmentation**: Extracts nuclei using [cellpose](https://cellpose.readthedocs.io/en/latest/index.html) for H&E and [SOP](https://github.com/loic-lb/Unsupervised-Nuclei-Segmentation-using-Spatial-Organization-Priors) for IHC brighfield images.

## Installation

To install PrismToolBox, run the following command in your terminal:

```bash
pip install prismtoolbox
```

To install PrismToolBox embedding and nuclei utilitaries, please run the following command:

```bash
pip install 'prismtoolbox[emb, seg]'
```

## Usage

Here's a basic example of how to use PrismToolBox:

```python
from prismtoolbox import WSI

# Initialize the reader
WSI_object = WSI(slide_path="path_to_your_slide", engine="openslide")

# Extract tissue contours
params_detect_tissue = {"seg_level": 4, "window_avg": 30, "window_eng": 5, "thresh": 90, "area_min": 1.5e3}

WSI_object.detect_tissue(**params_detect_tissue)

# Extract patches and save them as jpg images
params_patches = {"patch_size": 256, "patch_level": 0,"mode": "contours", "contours_mode": "four_pt_hard"}

WSI_object.extract_patches(**params_patches)
WSI_object.save_patches("path_to_folder", file_format="jpg")
```

For more detailed usage instructions and examples, please refer to the [documentation](https://gustaveroussy.github.io/prismtoolbox/).

## Contributing

Contributions to PrismToolBox are welcome! Please open an issue or submit a pull request to propose changes or additions.

## License

PrismToolBox is licensed under the BSD 3-Clause License. See [LICENSE](./LICENSE) for more information.
