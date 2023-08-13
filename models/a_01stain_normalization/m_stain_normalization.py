import sys
sys.path.append('../')

from tiatoolbox import data
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools import stainnorm
from pprint import pprint
import matplotlib.pyplot as plt
import argparse
import faulthandler
faulthandler.enable()

def wsi_stain_normalization(method, source, target=None):
    """
    Args:
        method (str): method of stain normalization
        source (np.array): source tile 
        target (np.array): target tile
    Returns:
        source (np.array): source tile
        normed_source (np.array): normalized source tile
        target (np.array): target tile
    """
    if target is None:
        target = data.stain_norm_target()
    stain_normalizer = stainnorm.get_normalizer(method)
    stain_normalizer.fit(target)
    normed_source = stain_normalizer.transform(source.copy())
    return source, normed_source, target

def plot(source, normed_source, target):
    plt.figure(figsize=(10, 5))
    plt.subplot(1,3,1)
    plt.imshow(source)
    plt.title("Source Image")
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.imshow(normed_source)
    plt.title("Normed Image")
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(target)
    plt.title("Target Image")
    plt.axis("off")
    plt.savefig("a_01stain_normalization/stain_normalization.jpg")


if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', default="/well/rittscher/shared/datasets/KiBla/cases/3923_21/3923_21_G_HE.isyntax")
    parser.add_argument('--tile_location', default=[50000, 50000], type=list)
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument('--tile_size', default=[1024, 1024], type=list)
    parser.add_argument('--stain_method', default='reinhard', help='method of stain normalization')
    args = parser.parse_args()

    ## read a WSI from isyntax
    wsi = WSIReader.open(args.slide_path)
    pprint(wsi.info.as_dict())
    source = wsi.read_region(args.tile_location, args.level, args.tile_size)
    # source = wsi.slide_thumbnail(resolution=1.25, units="power")
    source, normed_source, target = wsi_stain_normalization(args.stain_method, source)
    plot(source, normed_source, target)