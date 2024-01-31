import sys
sys.path.append('../')

import cv2
import pathlib
import logging
import json
import joblib
import argparse
import numpy as np

import matplotlib.pyplot as plt
import argparse
import numpy as np

from pprint import pprint
from common.m_utils import mkdir, load_json
from tiatoolbox.wsicore.wsireader import WSIReader, VirtualWSIReader, WSIMeta
from tiatoolbox.tools import patchextraction
from tiatoolbox.utils.misc import imwrite, imread
from models.a_01stain_normalization.m_stain_normalization import wsi_stain_normalization


def prepare_annotation_reader(wsi_path, wsi_ann_path, lab_dict, resolution=0.25, units="mpp"):
    """prepare annotation reader
    Args:
        wsi_path (str): path of a wsi of a tile
        wsi_ann_path (str): path of the annotation of the wsi or tile, should be in json format
        lab_dict (dict): a dict defines the label names and values, the label names should be the same
            as that in annotation file
        resolution (int): the resolution of preparing image
        units (str): the units of resolution, e.g., mpp
    Returns:
        wsi_reader (WSIReader): class::obj of reading wsi
        ann_reader (WSIReader): class::obj of reading annotation
        annotation (dict): loaded annotation file in dict format
    """
    assert units == "mpp", f"units should be mpp, but is {units}"
    if pathlib.Path(wsi_path).suffix == ".jpg":
        img = imread(wsi_path)
        metadata = WSIMeta(
            mpp=np.array([resolution, resolution]),
            objective_power=40,
            axes="YXS",
            slide_dimensions=np.array(img.shape[:2][::-1]),
            level_downsamples=[1.0],
            level_dimensions=[np.array(img.shape[:2][::-1])],
            raw={"xml": None},
        )
        wsi_reader = VirtualWSIReader(img, info=metadata)
    else:
        wsi_reader = WSIReader.open(wsi_path)
    annotation = load_json(wsi_ann_path)
    wsi_shape = wsi_reader.slide_dimensions(resolution=0, units="level")
    msk_readers = {}
    for k in list(lab_dict.keys())[1:]:
        polygons = annotation[k]["points"]
        if len(polygons) > 0:
            img = np.zeros((wsi_shape[1], wsi_shape[0]), np.uint8)
            for polygon in polygons:
                # polygon at given ann resolution
                if len(polygon) > 0:
                    polygon = np.array(polygon).astype(np.int32)
                    cv2.drawContours(img, [polygon], 0, 1, -1)

            # imwrite(f"{k}.jpg".replace("/", ""), img*255)
            msk_reader = VirtualWSIReader(img, info=wsi_reader.info, mode="bool")
            msk_readers.update({k: msk_reader})
    return wsi_reader, msk_readers, annotation


def generate_tile_from_wsi(
        wsi_paths, 
        wsi_ann_paths, 
        lab_dict, 
        save_tile_dir, 
        tile_size=None, 
        resolution=0.25, 
        units="mpp"
    ):
    """generate tiles from wsi based on annotation
    Args:
        wsi_paths (list): a list of wsi paths
        wsi_ann_paths (list): a list of annotation paths, should be one-to-one with wsi_paths
        lab_dict (dict): a dict defines the label names and values, the label names should be the same
            as that in annotation file
        save_tile_dir (str): directory of saving tiles
        tile_size (int or None): if none, extract tile from bounds of annotation
        resolution (int): the resolution of preparing tiles
        units (str): the units of resolution, e.g., mpp
    """
    mkdir(save_tile_dir)

    def _extract_bboxes(bbox, tile_size, scale):
        assert len(scale) == 2, f"length of scale should be 2, but is {len(scale)}"
        scale = [scale[0], scale[1], scale[0], scale[1]]
        bbox = [int(b * s) for b, s in zip(bbox, scale)]
        tile_size = int(tile_size)
        if tile_size is None:
            return [bbox]
        
        if bbox[2] - bbox[0] <= tile_size:
            x_start, x_end = (bbox[0] + bbox[2] - tile_size) // 2, bbox[2]
        else:
            x_start, x_end = bbox[0], bbox[2] - tile_size

        if bbox[3] - bbox[1] <= tile_size:
            y_start, y_end = (bbox[1] + bbox[3] - tile_size) // 2, bbox[3]
        else:
            y_start, y_end = bbox[1], bbox[3] - tile_size
        bbox_list = []
        for x in range(x_start, x_end, tile_size):
            for y in range(y_start, y_end, tile_size):
                bbox_list.append([x, y, x + tile_size, y + tile_size])
        return bbox_list

    def _extract_tile_from_wsi(idx, wsi_path, ann_path):
        logging.info("generating tiles from wsi: {}/{}...".format(idx + 1, len(wsi_paths)))
        wsi_name = pathlib.Path(wsi_path).stem
        wsi_reader, ann_readers, annotation = prepare_annotation_reader(wsi_path, ann_path, lab_dict, resolution, units)
        if len(ann_readers) > 0:
            for k in ann_readers.keys():
                ann_reader = ann_readers[k]
                bounds = annotation[k]["bounds"]
                # filter empty bounds
                bounds = [bbox for bbox in bounds if bbox[2] - bbox[0] > 0 and bbox[3] - bbox[1] > 0]
                # scale to give resolution
                assert units == "mpp", "units should be mpp"
                scale = wsi_reader.info.mpp / resolution
                for j, bound in enumerate(bounds):
                    bboxes = _extract_bboxes(bound, tile_size, scale)
                    t = 0
                    for bbox in bboxes:
                        tile = wsi_reader.read_bounds(bbox, resolution, units, coord_space="resolution")
                        anno = ann_reader.read_bounds(bbox, resolution, units, coord_space="resolution")
                        _, anno = cv2.threshold(anno, 0, 1, cv2.THRESH_BINARY)
                        if np.mean(anno) > 0.1:
                            contours, _ = cv2.findContours(anno, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                            polygons = [np.array(polygon).squeeze().tolist() for polygon in contours]
                            ann_dict = {k : {"bounds": [], "points": []} for k in list(lab_dict.keys())[1:]}
                            ann_bbox = [0, 0, anno.shape[0], anno.shape[1]]
                            ann_dict.update({k: {"bounds": [ann_bbox], "points": polygons}})
                            save_tile_path = pathlib.Path(save_tile_dir) / f"{wsi_name}.class{lab_dict[k]}.object{j}.tile{t}.jpg"
                            save_anno_path = pathlib.Path(save_tile_dir) / f"{wsi_name}.class{lab_dict[k]}.object{j}.annotation{t}.json"
                            logging.info(f"saving tile {wsi_name}.class{lab_dict[k]}.object{j}.tile{t}.jpg")
                            imwrite(save_tile_path, tile)
                            logging.info(f"saving annotation {wsi_name}.class{lab_dict[k]}.object{j}.annotation{t}.json")
                            with save_anno_path.open("w") as handle:
                                json.dump(ann_dict, handle) 
                            t = t + 1
        return
    
    # extract tile in parallel
    joblib.Parallel(n_jobs=8)(
        joblib.delayed(_extract_tile_from_wsi)(idx, wsi_path, ann_path)
        for idx, (wsi_path, ann_path) in enumerate(zip(wsi_paths, wsi_ann_paths))
    )
        
    return

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
    plt.figure(figsize=(5, 5))
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
    parser.add_argument('--slide_path', default="/well/rittscher/shared/datasets/KiBla/cases/3923_21/3923_21_G_HE.isyntax")
    parser.add_argument('--tile_location', default=[50000, 50000], type=list)
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument('--tile_size', default=[1024, 1024], type=list)
    parser.add_argument('--patch_size', default=[64, 64], type=list)
    parser.add_argument('--extract_method', default='slidingwindow', help='method of patch extraction')
    args = parser.parse_args()

    ## read a WSI from isyntax file
    wsi = WSIReader.open(args.slide_path)
    pprint(wsi.info.as_dict())
    image = wsi.read_region(args.tile_location, args.level, args.tile_size)
    _, image, _ = wsi_stain_normalization("reinhard", image)
    patches = wsi_patch_extraction(args.extract_method, image, args.patch_size)
    plot(patches)