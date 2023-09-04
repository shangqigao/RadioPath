import sys
sys.path.append('../')

from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.misc import imwrite
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.stainnorm import get_normalizer
from tiatoolbox.data import stain_norm_target
from tiatoolbox.utils.visualization import overlay_prediction_contours
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import os, argparse, joblib, random, cv2, shutil

from models.a_01stain_normalization.m_stain_normalization import wsi_stain_normalization

target_image = stain_norm_target()
stain_normaliser = get_normalizer("reinhard")
stain_normaliser.fit(target_image)


def stain_norm_func(img):
    return stain_normaliser.transform(img)

def rmdir(dir_path: str):
    """Remove a directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return

def wsi_nucleus_segmentation(wsi_path, save_dir, pretrained_model, tissue_masking, mode):
    """
    Args:
        wsi_path (str): path of wsi file
        save_dir (str): dir of saving segmentations
        pretrained_model (str): name of pretrained model
        tissue_masking (bool): if true, do tissue masking before patch extraction
        mode (str): "tile" or "wsi"
    Returns:
        wsi_output (list): a list of (wsi_path, output_path)
    """
    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model=pretrained_model,
        num_loader_workers=4,
        num_postproc_workers=4,
        batch_size=32,
        auto_generate_mask=tissue_masking,
        verbose=False,
    )
    inst_segmentor.model.preproc_func = stain_norm_func

    rmdir(save_dir)
    wsi_output = inst_segmentor.predict(
        [wsi_path],
        masks=None,
        save_dir=save_dir,
        mode=mode,
        on_gpu=True,
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

    plt.figure(figsize=(14, 5))
    plt.subplot(1,2,1)
    plt.imshow(wsi_overview)
    plt.axis("off")
    plt.title("WSI Overview")
    ax = plt.subplot(1,2,2)
    plt.imshow(overlaid_predictions)
    plt.axis("off")
    plt.title("Nucleus Segmentation")

    labels = [value[0] for value in color_dict.values()]
    colors = ["#%02x%02x%02x" % value[1] for value in color_dict.values()]
    markers = [plt.Line2D([0,0],[0,0],color=c, marker='o', linestyle='') for c in colors]
    ax.legend(markers, labels, fontsize="8", loc="center left", bbox_to_anchor=(1, 0.5))
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
    parser.add_argument('--slide_path', default="/well/rittscher/shared/datasets/KiBla/cases/3923_21/3923_21_G_HE.isyntax")
    parser.add_argument('--tile_location', default=[50000, 50000], type=list)
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument('--tile_size', default=[1024, 1024], type=list)
    parser.add_argument('--save_dir', default="a_04feature_extraction/wsi_nucleus_results", type=str)
    parser.add_argument('--pretrained_model', default="hovernet_fast-pannuke", type=str)
    parser.add_argument('--tissue_masking', default=True, type=bool)
    parser.add_argument('--mode', default="tile", type=str)
    args = parser.parse_args()

    ## read a WSI from isyntax
    wsi = WSIReader.open(args.slide_path)
    pprint(wsi.info.as_dict())
    if args.mode == "tile":
        tile = wsi.read_region(args.tile_location, args.level, args.tile_size)
        _, norm_tile, _ = wsi_stain_normalization("reinhard", tile)
        tile_save_path = os.path.join("a_04feature_extraction", 'tile_sample.png')
        imwrite(tile_save_path, tile)
        tile_output = wsi_nucleus_segmentation(tile_save_path, args.save_dir, args.pretrained_model, args.tissue_masking, args.mode)
        plot_wsi(norm_tile, tile_output)
    elif args.mode == "wsi":
        wsi_overview = wsi.slide_thumbnail(resolution=1.25, units="power")
        wsi_output = wsi_nucleus_segmentation(args.slide_path, args.save_dir, args.pretrained_model, args.tissue_masking, args.mode)
        plot_wsi(wsi_overview, wsi_output)
        plot_patches(wsi, wsi_output)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")