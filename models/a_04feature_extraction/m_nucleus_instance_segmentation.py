import sys
sys.path.append('../')

from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.misc import imread
from tiatoolbox.wsicore.wsireader import WSIReader

from tiatoolbox.utils.visualization import overlay_prediction_contours

import numpy as np
import matplotlib.pyplot as plt
import os, glob, argparse, joblib, random, cv2, shutil

def rmdir(dir_path: str):
    """Remove a directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return

def wsi_nucleus_segmentation(wsi_path, save_dir, pretrained_model, tissue_masking):
    """
    Args:
        wsi_path (str): path of wsi file
        save_dir (str): dir of saving segmentations
        pretrained_model (str): name of pretrained model
        tissue_masking (bool): if true, do tissue masking before patch extraction
    Returns:
        wsi_output (list): a list of (wsi_path, output_path)
    """
    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model=pretrained_model,
        num_loader_workers=4,
        num_postproc_workers=4,
        batch_size=8,
        auto_generate_mask=tissue_masking,
        verbose=False,
    )

    rmdir(save_dir)
    wsi_output = inst_segmentor.predict(
        [wsi_path],
        masks=None,
        save_dir=save_dir,
        mode='wsi',
        on_gpu=False,
        crash_on_exception=True,
    )
    return wsi_output

def plot_wsi(wsi_overview, wsi_output):
    wsi_pred = joblib.load(f"{wsi_output[0][1]}.dat")
    print("Number of detected nuclei: {}".format(len(wsi_pred)))

    color_dict = {
        0: ("Neoplastic epithelial", (255, 0, 0)),
        1: ("Inflammatory", (255, 255, 0)),
        2: ("Connective", (0, 255, 0)),
        3: ("Dead", (0, 0, 0)),
        4: ("Non-neoplastic epithelial", (0, 0, 255)),
    }

    overlaid_predictions = overlay_prediction_contours(
        canvas=wsi_overview,
        inst_dict=wsi_pred,
        draw_dot=False,
        type_colours=color_dict,
        line_thickness=4,
    )

    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.imshow(wsi_overview)
    plt.axis("off")
    plt.title("WSI Overview")
    plt.subplot(1,2,2)
    plt.imshow(overlaid_predictions)
    plt.axis("off")
    plt.title("Nucleus Segmentation")
    plt.savefig("a_04feature_extraction/wsi_nucleus_segmentation.jpg")

def plot_patches(wsi, wsi_output, bb=128):
    wsi_pred = joblib.load(f"{wsi_output[0][1]}.dat")
    print("Number of detected nuclei: {}".format(len(wsi_pred)))
    nuc_id_list = list(wsi_pred.keys())

    color_dict = {
        0: ("Neoplastic epithelial", (255, 0, 0)),
        1: ("Inflammatory", (255, 255, 0)),
        2: ("Connective", (0, 255, 0)),
        3: ("Dead", (0, 0, 0)),
        4: ("Non-neoplastic epithelial", (0, 0, 255)),
    }

    plt.figure(figsize=(10, 5))
    for i in range(4):
        selected_nuc_id = nuc_id_list[random.randint(0, len(wsi_pred))]
        sample_nuc = wsi_pred[selected_nuc_id]
        cent = np.int32(sample_nuc["centroid"])
        contour = sample_nuc["contour"]
        contour -= (cent - bb // 2)
        nuc_patch = wsi.read_rect(cent - bb // 2, bb, resolution=0.25, units="mpp", coord_space="resolution")
        overlaid_patch = cv2.drawContours(nuc_patch.copy(), [contour], -1, (255, 255, 0), 2)
        plt.subplot(2, 4, i + 1)
        plt.imshow(nuc_patch)
        plt.axis("off")
        plt.subplot(2, 4, i + 5)
        plt.imshow(overlaid_patch)
        plt.axis("off")
        plt.title(color_dict[sample_nuc["type"]][0])
    plt.savefig("a_04feature_extraction/patch_nucleus_segmentation.jpg")


if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', default="/well/rittscher/shared/datasets/KiBla/cases/1019_19/1019_19_2_L2_HE.isyntax")
    parser.add_argument('--save_dir', default="a_04feature_extraction/wsi_nucleus_results", type=str)
    parser.add_argument('--pretrained_model', default="hovernet_fast-pannuke", type=str)
    parser.add_argument('--tissue_masking', default=True, type=bool)
    args = parser.parse_args()

    ## read a WSI from isyntax
    wsi = WSIReader.open(args.slide_path)
    print(wsi.info.as_dict())
    wsi_overview = wsi.slide_thumbnail(resolution=1.25, units="power")
    wsi_output = wsi_nucleus_segmentation(args.slide_path, args.save_dir, args.pretrained_model, args.tissue_masking)
    plot_wsi(wsi_overview, wsi_output)
    plot_patches(wsi, wsi_output)