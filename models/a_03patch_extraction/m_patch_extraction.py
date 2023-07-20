import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import argparse
import numpy as np

from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools import patchextraction

def wsi_patch_extraction(method, image, size, locations=None):
    """
    Args:
        method (str): method of extracting patches
        image (np.array): object to extract
        size (intPaire): size of a patch
        locations (list): locations of extraction
    Return:
        patch_extractor (list): a list of extracted patches
    """
    if method == "point":
        if locations is None:
            raise ValueError("locations is required")
        patch_extractor = patchextraction.get_patch_extractor(
            input_img = image,
            locations_list = np.array(locations),
            method_name = "point",
            patch_size = size,
            resolution = 0,
            units = "level",
        )
    elif method == "slidingwindow":
        patch_extractor = patchextraction.get_patch_extractor(
            input_img = image,
            method_name = "slidingwindow",
            patch_size = size,
            stride = size,
        )
    else:
        raise ValueError("Invalid method for patch extraction")
    return patch_extractor

def plot(patches):
    i = 1
    plt.figure(figsize=(10, 5))
    for patch in patches:
        plt.subplot(4, 4, i)
        plt.imshow(patch)
        plt.axis("off")
        if i >= 16:
            break
        i += 1
    plt.savefig("a_03patch_extraction/patch_extraction.jpg")


if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', default="/well/rittscher/shared/datasets/KiBla/cases/1019_19/1019_19_2_L2_HE.isyntax")
    parser.add_argument('--tile_location', default=[40000, 50000], type=list)
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument('--tile_size', default=[1024, 1024], type=list)
    parser.add_argument('--patch_size', default=[64, 64], type=list)
    parser.add_argument('--extract_method', default='slidingwindow', help='method of patch extraction')
    args = parser.parse_args()

    ## read a WSI from isyntax file
    wsi = WSIReader.open(args.slide_path)
    image = wsi.read_region(args.tile_location, args.level, args.tile_size)
    patches = wsi_patch_extraction(args.extract_method, image, args.patch_size)
    plot(patches)