import sys
sys.path.append('../')

import random
import torch
import shutil
import os
import pathlib
import joblib
import argparse
import json
import pathlib
from pathlib import Path

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tiatoolbox.models import DeepFeatureExtractor, IOSegmentorConfig, NucleusInstanceSegmentor
from tiatoolbox.models.architecture.vanilla import CNNBackbone, CNNModel
from tiatoolbox.models.architecture.hipt import get_vit256
from tiatoolbox.tools.stainnorm import get_normalizer
from tiatoolbox.data import stain_norm_target
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.utils.misc import imwrite

from shapely.geometry import box as shapely_box
from shapely.strtree import STRtree
from pprint import pprint

SEED = 5
random.seed(SEED)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

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

def load_json(path: str):
    path = pathlib.Path(path)
    with path.open() as fptr:
        return json.load(fptr)

def recur_find_ext(root_dir: Path, exts):
    """Recursively find files with an extension in `exts`.

    This is much faster than glob if the folder
    hierachy is complicated and contain > 1000 files.

    Args:
        root_dir (Path):
            Root directory for searching.
        exts (list):
            List of extensions to match.

    Returns:
        List of full paths with matched extension in sorted order.

    """
    assert isinstance(exts, list)
    file_path_list = []
    for cur_path, _dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = Path(file_name).suffix
            if file_ext in exts:
                full_path = f"{cur_path}/{file_name}"
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list

def select_checkpoints(
    stat_file_path: str,
    top_k: int = 2,
    metric: str = "infer-valid-auprc",
    epoch_range = None,
):
    """Select checkpoints basing on training statistics.

    Args:
        stat_file_path (str): Path pointing to the .json
            which contains the statistics.
        top_k (int): Number of top checkpoints to be selected.
        metric (str): The metric name saved within .json to perform
            selection.
        epoch_range (list): The range of epochs for checking, denoted
            as [start, end] . Epoch x that is `start <= x <= end` is
            kept for further selection.

    Returns:
        paths (list): List of paths or info tuple where each point
            to the correspond check point saving location.
        stats (list): List of corresponding statistics.

    """
    if epoch_range is None:
        epoch_range = [0, 1000]
    stats_dict = load_json(stat_file_path)
    # k is the epoch counter in this case
    stats_dict = {
        k: v
        for k, v in stats_dict.items()
        if int(k) >= epoch_range[0] and int(k) <= epoch_range[1]
    }
    stats = [[int(k), v[metric], v] for k, v in stats_dict.items()]
    # sort epoch ranking from largest to smallest
    stats = sorted(stats, key=lambda v: v[1], reverse=True)
    # stats = stats[::-1]
    chkpt_stats_list = stats[:top_k]  # select top_k

    model_dir = Path(stat_file_path).parent
    epochs = [v[0] for v in chkpt_stats_list]
    paths = [
        (
            f"{model_dir}/epoch={epoch:03d}.extractor.weights.pth",
            f"{model_dir}/epoch={epoch:03d}.classifier.weights.pth",
        )
        for epoch in epochs
    ]
    chkpt_stats_list = [[v[0], v[2]] for v in chkpt_stats_list]
    print(paths)  # noqa: T201
    return paths, chkpt_stats_list

class CNNClassifier(CNNModel):
    def __init__(self, backbone, num_classes=1):
        super().__init__(backbone, num_classes)
        self._transform = self.transform()
    
    def forward(self, imgs):
        feat = self.feat_extract(imgs)
        gap_feat = self.pool(feat)
        return torch.flatten(gap_feat, 1)

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        image = batch_data.to(device).type(torch.float32)
        model.eval()
        with torch.inference_mode():
            output = model(image)
        return [output.cpu().numpy()]
 
    def postproc_func(self, output: np.ndarray):
        return output
    
    def preproc_func(self, image: np.ndarray):
        return self._transform(image=image)["image"]
        

    def load(self, feature_path, classifier_path):
        feature_state_dict = torch.load(feature_path)
        self.feat_extract.load_state_dict(feature_state_dict)
        classifier_state_dict = torch.load(classifier_path)
        self.classifier.load_state_dict(classifier_state_dict)

    def transform(self):
        TS = A.Compose([A.Normalize(), ToTensorV2()])
        return TS

def extract_cnn_features(wsi_paths, msk_paths, save_dir, mode, resolution=0.25, units="mpp"):
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": units, "resolution": resolution},],
        output_resolutions=[{"units": units, "resolution": resolution},],
        patch_input_shape=[224, 224],
        patch_output_shape=[224, 224],
        stride_shape=[224, 224],
        save_resolution={"units": "mpp", "resolution": 8.0}
    )
    
    # model = CNNBackbone("resnet50")
    model = CNNClassifier(backbone="resnet50", num_classes=1)
    pretrained_dir = "a_04feature_extraction/tile_bladder_models/cnn/00"
    stat_files = recur_find_ext(f"{pretrained_dir}/", [".json"])
    stat_files = [v for v in stat_files if ".old.json" not in v]
    assert len(stat_files) == 1
    pretrained, _ = select_checkpoints(
        stat_files[0],
        top_k=1,
        metric="infer-valid-A-auroc",
    )
    model.load(*pretrained[0])
    extractor = DeepFeatureExtractor(
        batch_size=32, 
        model=model, 
        num_loader_workers=8, 
    )

    rmdir(save_dir)
    output_map_list = extractor.predict(
        wsi_paths,
        msk_paths,
        mode=mode,
        ioconfig=ioconfig,
        on_gpu=True,
        crash_on_exception=True,
        save_dir=save_dir,
    )
    
    for input_path, output_path in output_map_list:
        input_name = pathlib.Path(input_path).stem
        output_parent_dir = pathlib.Path(output_path).parent

        src_path = pathlib.Path(f"{output_path}.position.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.position.npy")
        src_path.rename(new_path)

        src_path = pathlib.Path(f"{output_path}.features.0.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.features.npy")
        src_path.rename(new_path)

    return output_map_list

class ViT(torch.nn.Module):
    def __init__(self, model256_path):
        super().__init__()
        self._transform = self.transform()
        self.model256 = get_vit256(pretrained_weights=model256_path)
    
    def forward(self, imgs):
        feat = self.model256(imgs)
        return torch.flatten(feat, 1)

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        image = batch_data.to(device).type(torch.float32)
        model.eval()
        with torch.inference_mode():
            output = model(image)
        return [output.cpu().numpy()]
 
    def postproc_func(self, output: np.ndarray):
        return output
    
    def preproc_func(self, image: np.ndarray):
        return self._transform(image=image)["image"]

    def transform(self):
        TS = A.Compose([A.Normalize(), ToTensorV2()])
        return TS
    
def extract_vit_features(wsi_paths, msk_paths, save_dir, mode, resolution=0.25, units="mpp"):
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": units, "resolution": resolution},],
        output_resolutions=[{"units": units, "resolution": resolution},],
        patch_input_shape=[256, 256],
        patch_output_shape=[256, 256],
        stride_shape=[256, 256],
        save_resolution={"units": "mpp", "resolution": 8.0}
    )
    
    # model = CNNBackbone("resnet50")
    pretrained_path = "/well/rittscher/projects/shangqi-workspace/data/projects/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth"
    model = ViT(pretrained_path)
    extractor = DeepFeatureExtractor(
        batch_size=32, 
        model=model, 
        num_loader_workers=8, 
    )

    rmdir(save_dir)
    output_map_list = extractor.predict(
        wsi_paths,
        msk_paths,
        mode=mode,
        ioconfig=ioconfig,
        on_gpu=True,
        crash_on_exception=True,
        save_dir=save_dir,
    )
    
    for input_path, output_path in output_map_list:
        input_name = pathlib.Path(input_path).stem
        output_parent_dir = pathlib.Path(output_path).parent

        src_path = pathlib.Path(f"{output_path}.position.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.position.npy")
        src_path.rename(new_path)

        src_path = pathlib.Path(f"{output_path}.features.0.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.features.npy")
        src_path.rename(new_path)

    return output_map_list

def extract_composition_features(wsi_paths, msk_paths, save_dir, mode, resolution=0.25, units="mpp"):
    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        batch_size=16,
        num_loader_workers=8,
        num_postproc_workers=8,
    )
    if mode == "wsi":
        inst_segmentor.ioconfig.tile_shape = (5120, 5120)
    inst_segmentor.model.preproc_func = stain_norm_func

    rmdir(save_dir)
    output_map_list = inst_segmentor.predict(
        wsi_paths,
        msk_paths,
        mode=mode,
        on_gpu=True,
        crash_on_exception=True,
        save_dir=save_dir,
        resolution=resolution,
        units=units,
    )

    output_paths = []
    for input_path, output_path in output_map_list:
        input_name = pathlib.Path(input_path).stem
        output_parent_dir = pathlib.Path(output_path).parent

        src_path = pathlib.Path(f"{output_path}.dat")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.dat")
        src_path.rename(new_path)
        output_paths.append(new_path)
    for idx, path in enumerate(output_paths):
        if msk_paths is not None:
            get_cell_compositions(wsi_paths[idx], msk_paths[idx], path, save_dir, resolution=resolution, units=units)
        else:
            get_cell_compositions(wsi_paths[idx], None, path, save_dir, resolution=resolution, units=units)
    return output_paths


def get_cell_compositions(
        wsi_path,
        mask_path,
        inst_pred_path,
        save_dir,
        num_types = 2,
        patch_input_shape = (512, 512),
        stride_shape = (512, 512),
        resolution = 0.25,
        units = "mpp",
):
    if pathlib.Path(wsi_path).suffix == ".jpg":
        reader = WSIReader.open(wsi_path, mpp=(resolution, resolution))
    else:
        reader = WSIReader.open(wsi_path)
    inst_pred = joblib.load(inst_pred_path)
    inst_pred = {i: v for i, (_, v) in enumerate(inst_pred.items())}
    inst_boxes = [v["box"] for v in inst_pred.values()]
    inst_boxes = np.array(inst_boxes)

    geometries = [shapely_box(*bounds) for bounds in inst_boxes]
    spatial_indexer = STRtree(geometries)
    wsi_shape = reader.slide_dimensions(resolution=resolution, units=units)

    (patch_inputs, _) = PatchExtractor.get_coordinates(
        image_shape=wsi_shape,
        patch_input_shape=patch_input_shape,
        patch_output_shape=patch_input_shape,
        stride_shape=stride_shape,
    )

    if mask_path is not None:
        mask_reader = WSIReader.open(mask_path)
        mask_reader.info = reader.info
        selected_coord_indices = PatchExtractor.filter_coordinates(
            mask_reader,
            patch_inputs,
            wsi_shape=wsi_shape,
            min_mask_ratio=0.5,
        )
        patch_inputs = patch_inputs[selected_coord_indices]

    bounds_compositions = []
    for bounds in patch_inputs:
        bounds_ = shapely_box(*bounds)
        indices = [geo for geo in spatial_indexer.query(bounds_) if bounds_.contains(geometries[geo])]
        insts = [inst_pred[v]["type"] for v in indices]
        _, freqs = np.unique(insts, return_counts=True)
        holder = np.zeros(num_types, dtype=np.int16)
        holder[0] = freqs.sum()
        bounds_compositions.append(holder)
    bounds_compositions = np.array(bounds_compositions)

    base_name = pathlib.Path(wsi_path).stem
    np.save(f"{save_dir}/{base_name}.position.npy", patch_inputs)
    np.save(f"{save_dir}/{base_name}.features.npy", bounds_compositions)


if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', default="/well/rittscher/shared/datasets/KiBla/cases/3923_21/3923_21_G_HE.isyntax")
    parser.add_argument('--mask_method', default='otsu', help='method of tissue masking')
    parser.add_argument('--tile_location', default=[50000, 50000], type=list)
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument('--tile_size', default=[1024, 1024], type=list)
    parser.add_argument('--save_dir', default="a_04feature_extraction/wsi_features", type=str)
    parser.add_argument('--mode', default="tile", type=str)
    parser.add_argument('--feature_mode', default="cnn", type=str)
    args = parser.parse_args()

    wsi = WSIReader.open(args.slide_path)
    mask = wsi.tissue_mask(method=args.mask_method, resolution=1.25, units="power")
    pprint(wsi.info.as_dict())
    if args.mode == "tile":
        tile = wsi.read_region(args.tile_location, args.level, args.tile_size)
        wsi_path = os.path.join("a_04feature_extraction", 'tile_sample.jpg')
        imwrite(wsi_path, tile)
        tile_mask = mask.read_region(args.tile_location, args.level, args.tile_size)
        msk_path = os.path.join("a_04feature_extraction", 'tile_mask.jpg')
        imwrite(msk_path, np.uint8(tile_mask*255))
    elif args.mode == "wsi":
        wsi_mask = mask.slide_thumbnail(resolution=1.25, units="power")
        msk_path = os.path.join("a_04feature_extraction", 'wsi_mask.jpg')
        imwrite(msk_path, wsi_mask)
        wsi_path = args.slide_path
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    wsi_feature_dir = os.path.join(args.save_dir, args.feature_mode)
    if args.feature_mode == "composition":
        output_list = extract_composition_features(
            [wsi_path],
            [msk_path],
            wsi_feature_dir,
            args.mode,
        )
    elif args.feature_mode == "cnn":
        output_list = extract_cnn_features(
            [wsi_path],
            [msk_path],
            wsi_feature_dir,
            args.mode,
        )
    else:
        raise ValueError(f"Invalid feature mode: {args.feature_mode}")






