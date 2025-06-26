import os
import typer
import logging

from enum import Enum
from typing import Annotated
from pathlib import Path

from .utils import load_config_file

log = logging.getLogger(__name__)

app_preprocessing = typer.Typer(
    name="PrismToolBox Preprocessing CLI",
    help="A CLI for preprocessing WSIs using the PrismToolBox.",
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich")

class Engine(str, Enum):
    openslide = "openslide"
    tiffslide = "tiffslide"
    
class ContoursExtension(str, Enum):
    geojson = "geojson"
    pickle = "pickle"
    
class PatchExtension(str, Enum):
    h5 = "h5"
    geojson = "geojson"
    
class PatchMode(str, Enum):
    contours = "contours"
    roi = "roi"
    all = "all"
    
@app_preprocessing.command(no_args_is_help=True)
def contouring(
    slide_directory: Annotated[str, typer.Argument(
        help="Path to the directory containing the files.")],
    results_directory: Annotated[str, typer.Argument(
        help="Path to the directory where the results will be saved.")],
    engine: Annotated[Engine, typer.Option(
        help="Engine to use for reading the slides.",
        case_sensitive=False)] = Engine.openslide,
    annotations_directory: Annotated[str | None, typer.Option(
        help="Path to the directory containing the annotations."
    )] = None,
    contours_exts: Annotated[list[ContoursExtension], typer.Option(
        help="File extension for the contours annotations.",
        case_sensitive=False,
    )] = [ContoursExtension.pickle],
    config_file: Annotated[str, typer.Option(
        help="Path to the configuration file for tissue extraction."
    )] = "./config_contouring.yaml",
    visualize: Annotated[bool, typer.Option(
        help="Visualize the contours extracted.",
    )] = False,
):
    """Extract tissue contours from the slides in a specified directory."""
    import prismtoolbox as ptb
    
    # Set default parameters
    params_detect_tissue = {
        "seg_level": 2,
        "window_avg": 30,
        "window_eng": 3,
        "thresh": 120,
        "area_min": 6e3,
    }
    
    params_visualize_WSI = {
        "vis_level": 2,
        "number_contours": False,
        "line_thickness": 50,
    }
    
    if not os.path.exists(config_file):
        log.info(f"Using default parameters for tissue extraction.")
    else:
        log.info(f"Using parameters from config file: {config_file}")
        # Load parameters from config file
        params_detect_tissue = load_config_file(config_file, 
                                                dict_to_update=params_detect_tissue,
                                                key_to_check='contouring')
        if visualize:
            params_visualize_WSI = load_config_file(config_file,
                                                dict_to_update=params_visualize_WSI,
                                                key_to_check='visualizing')
        
    directory_contours = os.path.join(results_directory, f"contours")    
    Path(directory_contours).mkdir(parents=True, exist_ok=True)
    
    if visualize:
        directory_visualize = os.path.join(results_directory, f"contoured_images")
        Path(directory_visualize).mkdir(parents=True, exist_ok=True)

    # Iterate over the files in the directory
    for file_name in os.listdir(slide_directory):
        # Load the image
        WSI_object = ptb.WSI(os.path.join(slide_directory, file_name), engine=engine)
        print(f"Processing {WSI_object.slide_name}...")

        if f"{WSI_object.slide_name}.pkl" in os.listdir(directory_contours):
            continue
        
        # Extract the contours from the image
        WSI_object.detect_tissue(**params_detect_tissue)
        # Apply pathologist annotations
        if annotations_directory is not None:
            WSI_object.apply_pathologist_annotations(os.path.join(annotations_directory, f"{WSI_object.slide_name}.geojson"))
        # Save extracted contours
        for contours_ext in contours_exts:
            WSI_object.save_tissue_contours(directory_contours, file_format=contours_ext)
        if visualize:
            # Visualize the extracted contours on the tissue
            img = WSI_object.visualize(**params_visualize_WSI)
            img.save(os.path.join(directory_visualize, f"{WSI_object.slide_name}.jpg"))
    print("Contours extracted and saved successfully.")
    
@app_preprocessing.command(no_args_is_help=True)
def patching(
    slide_directory: Annotated[str, typer.Argument(
        help="Path to the directory containing the files.")],
    results_directory: Annotated[str, typer.Argument(
        help="Path to the directory where the results will be saved.")],
    contours_directory: Annotated[str | None, typer.Option(
        help="Path to the directory containing the contours annotations."
    )] = None,
    engine: Annotated[Engine, typer.Option(
        help="Engine to use for reading the slides.",
        case_sensitive=False)] = Engine.openslide,
    patch_exts: Annotated[list[PatchExtension], typer.Option(
        help="File extension for the patches.",
        case_sensitive=False,
    )] = [PatchExtension.h5],
    mode: Annotated[PatchMode, typer.Option(
        help="The mode to use for patch extraction. Possible values are 'contours', 'roi', and 'all'.",
        case_sensitive=False,
    )] = PatchMode.contours,
    config_file: Annotated[str, typer.Option(
        help="Path to the configuration file for patch extraction."
    )] = "./config_patching.yaml",
    stitch: Annotated[bool, typer.Option(
        help="Whether to stitch the extracted patches into a single image for visualization.",
    )] = True,
):
    """Extract patches from the slides in a specified directory."""
    import prismtoolbox as ptb
    assert mode == "contours" and contours_directory is not None, \
        "If the mode is 'contours', you must provide a directory with contours annotations. Please use the `contouring` command to extract contours first."
    
    # Set default parameters
    params_patches = {"patch_size": 256, "patch_level": 0, "overlap": 0,
                      "units": ["px", "px"], "contours_mode": "four_pt", "rgb_threshs": [2, 240], "percentages": [0.6, 0.9]}
    params_stitch_WSI = {"vis_level": 2, "draw_grid": False}
    
    if not os.path.exists(config_file):
        log.info(f"Using default parameters for tissue extraction.")
    else:
        log.info(f"Using parameters from config file: {config_file}")
        # Load parameters from config file
        params_patches = load_config_file(config_file, 
                                        dict_to_update=params_patches,
                                        key_to_check='patching')
        if stitch:
            params_stitch_WSI = load_config_file(config_file,
                                                dict_to_update=params_stitch_WSI,
                                                key_to_check='stitching')
    
    assert mode == "contours" and contours_directory is not None, \
    "If the mode is 'roi' or 'all', you must provide a directory with contours annotations. " 
    "Please use the `contouring` command to extract contours first."
                        
    # Path to the directory where the patches will be saved
    directory_patches = os.path.join(results_directory,
                                     f"patches_{params_patches['patch_size']}_overlap"
                                     f"_{params_patches['overlap']}")
    Path(directory_patches).mkdir(parents=True, exist_ok=True)
    
    if stitch:
        # Path to the directory where the stitched images will be saved
        directory_stitch = os.path.join(results_directory,
                                        f"stitched_images_{params_patches['patch_size']}_overlap"
                                        f"_{params_patches['overlap']}")
        Path(directory_stitch).mkdir(parents=True, exist_ok=True)
        
    # Iterate over the files in the directory
    for file_name in os.listdir(slide_directory):
        # Load the image
        WSI_object = ptb.WSI(os.path.join(slide_directory, file_name), engine=engine)
        print(f"Processing {WSI_object.slide_name}...")

        if f"{WSI_object.slide_name}.h5" in os.listdir(directory_patches):
            continue
        
        # Load the contours for the image
        WSI_object.load_tissue_contours(os.path.join(contours_directory, f"{WSI_object.slide_name}.pkl"))

        if mode == "roi":
            WSI_object.set_roi()

        # Extract patches from the contours
        WSI_object.extract_patches(mode=mode, **params_patches)
        # Save the extracted patches
        for patch_ext in patch_exts:
            WSI_object.save_patches(directory_patches, file_format=patch_ext)
        if stitch:
            # Stitch the extracted patches
            if params_stitch_WSI["vis_level"] >= len(WSI_object.level_dimensions):
                continue
            img = WSI_object.stitch(**params_stitch_WSI)

            img.save(os.path.join(directory_stitch, f"{WSI_object.slide_name}.jpg"))
    print("Done !")