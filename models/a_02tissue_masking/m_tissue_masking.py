import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import argparse
import pathlib
import logging
import joblib
import random

import numpy as np
from common.m_utils import mkdir
from tiatoolbox.wsicore.wsireader import WSIReader, VirtualWSIReader, WSIMeta
from tiatoolbox.utils.misc import imwrite, imread
from tiatoolbox.tools import tissuemask
from pprint import pprint

def generate_wsi_tissue_mask(wsi_paths, save_msk_dir=None, method="otsu", n_jobs=8, resolution=1.25, units="power"):
    """generate tissue masks for a list of whole slide images
    Args:
        wsi_paths (List): a list of wsi paths
        save_msk_dir (str): directory of saving masks
        method (str): method of tissue masking
        resolution (float): resolution of extacting thumbnails
        units (str): units of resolution, e.g., mpp
    Returns:
        If not save masks, return a fitted masker
    """
    def _extract_wsi_thumbs(path):
        if pathlib.Path(path).suffix == ".jpg":
            img_name = pathlib.Path(path).stem
            logging.info(f"Reading image: {img_name}...")
            img = imread(path)
            metadata = WSIMeta(
                mpp=np.array([0.5, 0.5]),
                objective_power=20,
                axes="YXS",
                slide_dimensions=np.array(img.shape[:2][::-1]),
                level_downsamples=[1.0],
                level_dimensions=[np.array(img.shape[:2][::-1])],
                raw={"xml": None},
            )
            wsi = VirtualWSIReader(img, info=metadata)
        else:
            img_name = pathlib.Path(path).stem
            logging.info(f"Reading WSI: {img_name}...")
            wsi = WSIReader.open(path)
        wsi_thumb = wsi.slide_thumbnail(resolution=resolution, units=units)
        wsi_thumb = np.array(wsi_thumb, np.uint8)
        zeros = wsi_thumb.mean(axis=2) < 100
        wsi_thumb[zeros, :] = 255
        del wsi
        return wsi_thumb
    
    # extract wsi thumbnails in parallel
    wsi_thumbs = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_extract_wsi_thumbs)(path)
        for path in wsi_paths
    )

    if method not in ["otsu", "morphological"]:
            raise ValueError(f"Invalid tissue masking method: {method}.")
    if method == "morphological":
        mpp = None
        power = None
        if units == "mpp":
            mpp = resolution
        elif units == "power":
            power = resolution
        masker = tissuemask.MorphologicalMasker(mpp=mpp, power=power)
    elif method == "otsu":
        masker = tissuemask.OtsuTissueMasker()
    
    # fitting all images to compute threshold
    logging.info(f"Fitting {len(wsi_paths)} images...")
    masker.fit(wsi_thumbs)

    if save_msk_dir is not None:
        save_msk_dir = pathlib.Path(save_msk_dir)
        mkdir(save_msk_dir)
        for path, wsi_thumb in zip(wsi_paths, wsi_thumbs):
            wsi_name = pathlib.Path(path).stem
            msk_thumb = masker.transform([wsi_thumb])[0]
            save_msk_path = save_msk_dir / f"{wsi_name}.jpg"
            logging.info(f"Saving tissue mask {wsi_name}.jpg")
            imwrite(save_msk_path, msk_thumb.astype(np.uint8)*255)
        return
    else:
        return masker

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
    parser.add_argument('--slide_path', default="/well/rittscher/shared/datasets/KiBla/cases/3923_21/3923_21_G_HE.isyntax")
    parser.add_argument('--tile_location', default=[0, 0], type=list)
    parser.add_argument('--level', default=1, type=int)
    parser.add_argument('--tile_size', default=[1024, 1024], type=list)
    parser.add_argument('--mask_method', default='morphological', help='method of tissue masking')
    args = parser.parse_args()

    ## read a WSI from isyntax file
    wsi = WSIReader.open(args.slide_path)
    pprint(wsi.info.as_dict())
    wsi_thumb = wsi.slide_thumbnail(resolution=1.25, units="power")
    mask = wsi.tissue_mask(method=args.mask_method, resolution=1.25, units="power")
    mask_thumb = mask.slide_thumbnail(resolution=1.25, units="power")
    plot(wsi_thumb, mask_thumb)