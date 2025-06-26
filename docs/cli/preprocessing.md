# CLI Reference: Preprocessing

The PrismToolBox CLI provides useful preprocessing capabilities for whole slide images through the `ptb preprocessing` command.

## Overview

The preprocessing module includes two main commands:

1. **`contouring`**: Extract tissue contours from whole slide images
2. **`patching`**: Extract patches from slides using tissue contours

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

### `ptb preprocessing contouring`

Extract tissue contours from whole slide images.

#### Usage

```bash
ptb preprocessing contouring [OPTIONS] SLIDE_DIRECTORY RESULTS_DIRECTORY
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
--8<-- "./config/default_contouring.yaml"
```

#### Examples

```bash
# Basic contour extraction
ptb preprocessing contouring slides/ results/

# With visualization
ptb preprocessing contouring slides/ results/ --visualize

# Using custom configuration
ptb preprocessing contouring slides/ results/ --config-file custom_config.yaml

# With annotations and multiple output formats
ptb preprocessing contouring slides/ results/ --annotations-directory annotations/ --contours-exts pickle geojson --visualize
```

### `ptb preprocessing patching`

Extract patches from slides using tissue contours.

#### Usage

```bash
ptb preprocessing patching [OPTIONS] SLIDE_DIRECTORY RESULTS_DIRECTORY
```

#### Arguments

- **`SLIDE_DIRECTORY`**: Path to the directory containing the slide files
- **`RESULTS_DIRECTORY`**: Path to the directory where the results will be saved

#### Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `--contours-directory` | `str | None` | Path to directory containing contour annotations | `None` |
| `--engine` | `str` | Engine for reading slides | `openslide` |
| `--mode` | `str` | Extraction mode (`contours`, `roi`, `all`) | `contours` |
| `--patch-exts` | `list[str]` | File extensions for patches (`h5`, `geojson`) |`[h5]` |
| `--config-file` | `str | None` | Path to configuration file | `None` |

#### Configuration File

Example configuration for patch extraction:

```yaml
--8<-- "./config/default_patching.yaml"
```

#### Examples

```bash
# Basic patch extraction
ptb preprocessing patching slides/ results/ --contours-directory results/contours/

# With custom configuration
ptb preprocessing patching slides/ results/  --contours-directory results/contours/ --config-file patch_config.yaml

# Extract patches in multiple formats
ptb preprocessing patching slides/ results/ --contours-directory results/contours/ --patch-exts h5 geojson
```

## Complete Workflow Example

Here's a complete example of processing a dataset:

```bash
# Step 1: Extract tissue contours with visualization
ptb preprocessing contouring slides/ results/ --visualize --config-file tissue_config.yaml

# Step 2: Extract patches from the contours
ptb preprocessing patching slides/ results/ --contours-directory results/contours/ --config-file patch_config.yaml --patch-exts geojson

# Results will be saved in:
# - results/contours/        (tissue contours)
# - results/contoured_images/ (visualizations)
# - results/patches_256_ovelap_0/       (extracted patches)
# - results/stitched_images_256_ovelap_0/ (patch visualizations)
```

## Error Handling

Common issues and solutions:

### Missing Dependencies

```bash
Error: Segmentation features require additional dependencies.
Please install with: pip install prismtoolbox[seg]
```

**Solution**: Install the required dependencies:
```bash
pip install prismtoolbox[seg,emb]
```

### Configuration File Issues

```bash
Warning: Incomplete tissue extraction parameters in config file
```

**Solution**: Ensure your configuration file contains all required parameters for each section.

### File Path Issues

```bash
Error: No valid config file found. Using default parameters.
```

**Solution**: Check that your configuration file path is correct and the file exists.

## Tips and Best Practices

1. **Start with visualization**: Use `--visualize` flag to check if tissue detection works correctly
2. **Test with small datasets**: Process a few slides first to validate your parameters
3. **Use configuration files**: Store your parameters in YAML files for reproducibility
4. **Monitor output**: Use verbose mode (`-v` or `-vv`) to see detailed processing information
