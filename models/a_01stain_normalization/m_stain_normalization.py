from tiatoolbox import data
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools import stainnorm
import matplotlib.pyplot as plt
import argparse

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
    plt.show()


if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', default="/well/rittscher/shared")
    parser.add_argument('--tile_location', default=[0, 0], type=list)
    parser.add_argument('--level', default=1, type=int)
    parser.add_argument('--tile_size', default=[1024, 1024], type=list)
    parser.add_argument('--stain_method', default='vahadane', help='method of stain normalization')
    args = parser.parse_args()

    ## read a WSI from isyntax
    wsi = WSIReader.open(args.slide_path)
    source = wsi.read_region(args.tile_location, args.level, args.tile_size)
    source, normed_source, target = wsi_stain_normalization(args.stain_method, source)
    plot(source, normed_source, target)