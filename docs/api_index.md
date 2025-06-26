# API Reference

Welcome to the PrismToolBox API documentation. This section provides comprehensive reference for all modules, classes, and functions in the library.

## Core Modules

### 🔬 WSI Core (`prismtoolbox.wsicore`)

The core module for whole slide image handling and preprocessing.

**Main Classes:**
- [`WSI`](reference/prismtoolbox/wsicore/wsi.md) - Primary class for WSI operations, tissue detection, patch extraction, and visualization

**Key Functionality:**
- Slide loading and multi-resolution access
- Tissue contour detection and analysis
- ROI (Region of Interest) management
- Patch extraction with various modes
- Visualization and stitching capabilities
- QuPath integration for annotations

---

### 🧠 WSI Embeddings (`prismtoolbox.wsiemb`)

Feature extraction and embedding generation from WSI patches.

**Main Classes:**
- [`SlideEmbedder`](reference/prismtoolbox/wsiemb/embedder.md) - Extract embeddings from slide patches using pretrained models
- [`PatchEmbedder`](reference/prismtoolbox/wsiemb/embedder.md) - Extract embeddings from patch datasets 
- [`EmbeddingProcessor`](reference/prismtoolbox/wsiemb/processing.md) - Process, analyze, and visualize embeddings

**Key Functionality:**
- Model-based embeddings (ResNet, Vision Transformers, Foundation models)
- Stain-based feature extraction (color deconvolution)
- Cell-based feature extraction (morphological features)
- Dimensionality reduction and clustering
- Visualization and analysis tools

---

### 🔬 Nuclei Segmentation (`prismtoolbox.nucleiseg`)

Deep learning-based nuclei segmentation and analysis.

**Main Classes:**
- [`NucleiSegmenter`](reference/prismtoolbox/nucleiseg/segmenter.md) - Segment nuclei in WSI patches using deep learning models

**Key Functionality:**
- Multiple segmentation models (SOP, custom models)
- Batch processing of slide patches
- Post-processing and conflict resolution
- QuPath export for visualization

---