import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import argparse

from tiatoolbox.wsicore.wsireader import WSIReader

def plot(image, mask):
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(mask)
    plt.title("Mask")
    plt.axis("off")
    plt.savefig("a_02tissue_masking/tissue_masking.jpg")


if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', default="/well/rittscher/shared/datasets/KiBla/cases/1019_19/1019_19_2_L2_HE.isyntax")
    parser.add_argument('--tile_location', default=[0, 0], type=list)
    parser.add_argument('--level', default=1, type=int)
    parser.add_argument('--tile_size', default=[1024, 1024], type=list)
    parser.add_argument('--mask_method', default='otsu', help='method of tissue masking')
    args = parser.parse_args()

    ## read a WSI from isyntax file
    wsi = WSIReader.open(args.slide_path)
    wsi_thumb = wsi.slide_thumbnail(resolution=1.25, units="power")
    mask = wsi.tissue_mask(method=args.mask_method, resolution=1.25, units="power")
    mask_thumb = mask.slide_thumbnail(resolution=1.25, units="power")
    plot(wsi_thumb, mask_thumb)