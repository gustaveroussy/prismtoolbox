---
hide:
  - navigation
  - toc
  - path
---

# PrismToolBox Documentation

**PrismToolBox** is a comprehensive Python library for histopathology image analysis, providing powerful tools for processing whole slide images (WSI), feature extraction, and nuclei segmentation.

## 🔬 What is PrismToolBox?

PrismToolBox is designed for researchers and practitioners working with digital pathology images. It offers a complete pipeline for:

- **WSI Preprocessing**: Handle tissue contouring, patch extraction, and visualization
- **Feature Extraction**: Extract embeddings from pretrained models from slide or patch datasets
- **Nuclei Segmentation**: Advanced deep learning models for cell detection and segmentation

## 🛠️ Installation

### Basic Installation

```bash
pip install prismtoolbox
```

### Installation with Optional Dependencies
```bash
pip install prismtoolbox[emb, seg]
```

### Development Installation

```bash
git clone https://github.com/gustaveroussy/PrismToolBox.git
cd PrismToolBox
pip install -e .
```

## 🚀 Key Features

### Core Modules

- **`wsicore`**: Core functionality for WSI handling and preprocessing
- **`wsiemb`**: Feature extraction
- **`nucleiseg`**: Deep learning-based nuclei segmentation

### Command Line Interface

PrismToolBox comes with a powerful CLI that makes it easy to process large datasets:

```bash
# Use the CLI
ptb --help
```

## 📋 Quick Start

### Python API

```python
from prismtoolbox import WSI

# Initialize the reader
WSI_object = WSI(slide_path="path_to_your_slide", engine="openslide")

# Extract tissue contours
params_detect_tissue = {"seg_level": 2, "window_avg": 30, "window_eng": 3, "thresh": 120, "area_min": 6e3}

WSI_object.detect_tissue(**params_detect_tissue)

# Extract patches and save them as jpg images
params_patches = {"patch_size": 256, "patch_level": 0, "overlap": 0, "contours_mode": "four_pt"}

WSI_object.extract_patches(**params_patches)
WSI_object.save_patches("path_to_folder", file_format="jpg")
```

### Command Line Interface

```bash
# Extract tissue contours
ptb preprocessing contouring slides/ results/ --visualize

# Extract patches
ptb preprocessing patching slides/ results/ --contours-directory results/contours/
```

## 📚 Documentation Structure

- **[API Reference](api_index.md)**: Detailed documentation of all classes and functions
- **[CLI Reference](cli/preprocessing.md)**: Complete guide to the command-line interface
- **[Examples](examples/index.md)**: Practical examples and tutorials

## 🤝 Contributing

We welcome contributions! Please see our [contribution guidelines](https://github.com/gustaveroussy/PrismToolBox/blob/main/CONTRIBUTING.md) for more information.

## 📄 License

This project is licensed under the BSD-3-Clause license - see the [LICENSE](https://github.com/gustaveroussy/prismtoolbox/blob/master/LICENSE) file for details.

## 🔗 Links

- [GitHub Repository](https://github.com/gustaveroussy/PrismToolBox)
- [PyPI Package](https://pypi.org/project/prismtoolbox/)
- [Issue Tracker](https://github.com/gustaveroussy/PrismToolBox/issues)

