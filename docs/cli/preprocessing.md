# CLI Reference: Preprocessing

The PrismToolBox CLI provides useful preprocessing capabilities for whole slide images through the `ptb preprocessing` command.

## Overview

The preprocessing module includes two main commands:

1. **`contour`**: Extract tissue contours from whole slide images
2. **`patchify`**: Extract patches from slides using tissue contours

## Installation

Make sure you have PrismToolBox installed:

```bash
# Basic installation
pip install prismtoolbox
```

## Global Options

All preprocessing commands support these global options:

- `--verbose, -v`: Increase verbosity (can be used multiple times: `-v`, `-vv`)
- `--help`: Show help message

## Commands

### `ptb preprocessing contour`

Extract tissue contours from whole slide images.

#### Usage

```bash
ptb preprocessing contour [OPTIONS] SLIDE_DIRECTORY RESULTS_DIRECTORY
```

#### Arguments

- **`SLIDE_DIRECTORY`**: Path to the directory containing the slide files
- **`RESULTS_DIRECTORY`**: Path to the directory where the results will be saved

#### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--engine` | `str` | Engine for reading slides (`openslide`, `tiffslide`). | `openslide` |
| `--annotations-directory` | `str | None` | Path to annotations directory | `None` |
| `--contours-exts` | `list[str]` | File extensions for contour annotations (`geojson`, `pickle`) | `[pickle]` |
| `--config-file` | `str`  | Path to configuration file | `None` |
| `--visualize` | `bool` | Visualize the extracted contours | `False` |

#### Configuration File

You can use a YAML configuration file to specify tissue extraction and visualization parameters:

```yaml
--8<-- "./config/default_contour_config.yaml"
```

#### Examples

```bash
# Basic contour extraction
ptb preprocessing contour slides/ results/

# With visualization
ptb preprocessing contour slides/ results/ --visualize

# Using custom configuration
ptb preprocessing contour slides/ results/ --config-file custom_config.yaml

# With annotations and multiple output formats
ptb preprocessing contour slides/ results/ --annotations-directory annotations/ --contours-exts pickle geojson --visualize
```

### `ptb preprocessing patchify`

Extract patches from slides using tissue contours.

#### Usage

```bash
ptb preprocessing patchify [OPTIONS] SLIDE_DIRECTORY RESULTS_DIRECTORY
```

#### Arguments

- **`SLIDE_DIRECTORY`**: Path to the directory containing the slide files
- **`RESULTS_DIRECTORY`**: Path to the directory where the results will be saved

#### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--roi-csv` | `str | None` | Path to the csv file containing the ROIs | `None` |
| `--contours-directory` | `str | None` | Path to directory containing contour annotations | `None` |
| `--engine` | `str` | Engine for reading slides | `openslide` |
| `--mode` | `str` | Extraction mode (`contours`, `roi`, `all`) | `contours` |
| `--patch-exts` | `list[str]` | File extensions for patches (`h5`, `geojson`) |`[h5]` |
| `--config-file` | `str | None` | Path to configuration file | `None` |

#### Configuration File

Example configuration for patch extraction:

```yaml
--8<-- "./config/default_patch_config.yaml"
```

#### Examples

```bash
# Basic patch extraction
ptb preprocessing patchify slides/ results/

# Patch extraction within a ROI
ptb preprocessing patchify slides/ results/ --mode roi  --roi-directory results/rois.csv

# Within previously extracted tissue contours and custom configuration
ptb preprocessing patchify slides/ results/ --mode contours  --contours-directory results/contours/ --config-file patch_config.yaml

# Extract patches in multiple formats
ptb preprocessing patchify slides/ results/ --contours-directory results/contours/ --patch-exts h5 geojson
```

Attention: For the `roi` mode, you need to provide a table with the ROIs in a CSV format, where each row corresponds to a slide and contains the slide ID and coordinates of the ROI.

```bash
# Extract patches from a specific ROI
ptb preprocessing patchify slides/ results/ --mode roi --roi-csv results/rois.csv
```

## Complete Workflow Example

Here's a complete example of processing a dataset:

```bash
# Step 1: Extract tissue contours with visualization
ptb preprocessing contouring slides/ results/ --visualize --config-file tissue_config.yaml

# Step 2: Extract patches from the contours
ptb preprocessing patching slides/ results/ --contours-directory results/contours/ --config-file patch_config.yaml --patch-exts geojson
```

Results will be saved in:
- results/contours/ (tissue contours as pickle files)
- results/contoured_images/ (visualizations)
- results/patches_256_ovelap_0/ (extracted patches as geojson coordinates)
- results/stitched_images_256_ovelap_0/ (patch visualizations)

## Tips and Best Practices

1. **Start with small datasets**: Process a few slides first to validate your parameters
2. **Use visualizations**: Use `--visualize` flag to check if tissue detection works correctly, and `--stitch` to visualize the selected patches.
3. **Monitor output**: Use verbose mode (`-v` or `-vv`) to see detailed processing information
