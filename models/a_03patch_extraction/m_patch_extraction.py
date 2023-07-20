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
        size (int): size of a patch
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
            method_name="point",
            patch_size = (size, size,),
            resolution=0,
            units="level",
        )
    elif method == "slidingwindow":
        patch_extractor = patchextraction.get_patch_extractor(
            input_img = image,
            method_name="slidingwindow",
            patch_size=(size, size,),
            stride=(size, size),
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
        if i > 16:
            break
        i += 1
    plt.show()


if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', default="/well/rittscher/shared")
    parser.add_argument('--tile_location', default=[0, 0], type=list)
    parser.add_argument('--level', default=1, type=int)
    parser.add_argument('--tile_size', default=[1024, 1024], type=list)
    parser.add_argument('--patch_size', default=[32, 32], type=list)
    parser.add_argument('--extract_method', default='slidingwindow', help='method of patch extraction')
    args = parser.parse_args()

    ## read a WSI from isyntax file
    wsi = WSIReader.open(args.slide_path)
    image = wsi.read_region(args.tile_location, args.level, args.tile_size)
    patches = wsi_patch_extraction(args.extract_method, image, args.patch_size)
    plot(patches)