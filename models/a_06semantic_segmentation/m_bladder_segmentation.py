import sys
sys.path.append('../')

import cv2
import os, glob
import pathlib
import logging
import random
import json
import copy
import torch
import tqdm
import joblib
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression as PlattScaling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score as auprc_scorer
from sklearn.metrics import roc_auc_score as auroc_scorer
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader

from tiatoolbox.utils.misc import save_as_json
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.utils.misc import imwrite
from tiatoolbox import logger

from models.a_04feature_extraction.m_feature_extraction import extract_composition_features
from models.a_04feature_extraction.m_feature_extraction import extract_deep_features
from models.a_05feature_aggregation.m_graph_construction import construct_graph
from models.a_05feature_aggregation.m_graph_construction import visualize_graph
from models.a_05feature_aggregation.m_graph_neural_network import SlideGraphArch
from models.a_05feature_aggregation.m_graph_neural_network import SlideGraphDataset
from models.a_05feature_aggregation.m_graph_neural_network import ScalarMovingAverage

torch.multiprocessing.set_sharing_strategy("file_system")

def mkdir(dir_path: Path):
    """Create a directory if it does not exist."""
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)
    return

def rm_n_mkdir(dir_path: Path):
    """Remove then re-create a directory."""
    if dir_path.is_dir():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True)
    return

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
                full_path = cur_path / file_name
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list

def load_json(path: str):
    with path.open() as fptr:
        return json.load(fptr)

def generate_wsi_tissue_mask(wsi_paths, save_msk_dir, method="otsu"):
    """generate tissue masks for a list of wsi"""
    save_msk_dir = pathlib.Path(save_msk_dir)
    mkdir(save_msk_dir)
    for path in wsi_paths:
        wsi_name = pathlib.Path(path).stem
        wsi = WSIReader.open(path)
        msk = wsi.tissue_mask(method, resolution=1.25, units="power")
        msk_thumb = msk.slide_thumbnail(resolution=1.25, units="power")
        save_msk_path = save_msk_dir / f"{wsi_name}.jpg"
        logging.info(f"Saving tissue mask {wsi_name}.jpg")
        imwrite(save_msk_path, np.uint8(msk_thumb*255))
    return

def prepare_annotation_reader(wsi_path, wsi_ann_path, lab_dict, ann_mpp, resolution, units):
    """prepare annotation reader"""
    if pathlib.Path(wsi_path).suffix == ".jpg":
        assert units == "mpp", "units must be mpp"
        wsi_reader = WSIReader.open(wsi_path, mpp=resolution)
    else:
        wsi_reader = WSIReader.open(wsi_path)
    annotation = load_json(wsi_ann_path)
    wsi_shape = wsi_reader.slide_dimensions(resolution=ann_mpp, units="mpp")
    msk_readers = {}
    for k in list(lab_dict.keys())[1:]:
        polygons = annotation[k]["points"]
        if len(polygons) > 0:
            img = np.zeros((wsi_shape[1], wsi_shape[0]), np.uint8)
            for polygon in polygons:
                # polygon at given ann resolution
                polygon = np.array(polygon) * wsi_reader.info.mpp / ann_mpp
                polygon = polygon.astype(np.int32)
                cv2.drawContours(img, [polygon], 0, 255, -1)

            # imwrite(f"{k}.jpg".replace("/", ""), img)
            msk_reader = WSIReader.open(img)
            msk_reader.info = wsi_reader.info
            msk_readers.update({k: msk_reader})
    return wsi_reader, msk_readers, annotation


def generate_tile_from_wsi(
        wsi_paths, 
        wsi_ann_paths, 
        lab_dict, 
        save_tile_dir, 
        ann_mpp, 
        tile_size=None, 
        resolution=0.25, 
        units="mpp"
    ):
    """generate tile from wsi based on annotation"""
    mkdir(save_tile_dir)
    for idx, (wsi_path, ann_path) in enumerate(zip(wsi_paths, wsi_ann_paths)):
        logging.info("generating tiles from wsi: {}/{}...".format(idx + 1, len(wsi_paths)))
        wsi_name = pathlib.Path(wsi_path).stem
        wsi_reader, ann_readers, annotation = prepare_annotation_reader(wsi_path, ann_path, lab_dict, ann_mpp, resolution, units)
        if len(ann_readers) > 0:
            for k in ann_readers.keys():
                ann_reader = ann_readers[k]
                bounds = annotation[k]["bounds"]
                for j, bbox in enumerate(bounds):
                    if tile_size is not None:
                        bbox = (bbox[0], bbox[1], tile_size, tile_size)
                    tile = wsi_reader.read_bounds(bbox, resolution, units)
                    anno = ann_reader.read_bounds(bbox, resolution, units)
                    anno = cv2.cvtColor(anno, cv2.COLOR_RGB2GRAY)
                    _, anno = cv2.threshold(anno, 1, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(anno, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    polygons = [np.array(polygon).squeeze().tolist() for polygon in contours]
                    ann_dict = {k : {"bounds": [], "points": []} for k in list(lab_dict.keys())[1:]}
                    ann_bbox = [0, 0, anno.shape[0], anno.shape[1]]
                    ann_dict.update({k: {"bounds": [ann_bbox], "points": polygons}})
                    save_tile_path = pathlib.Path(save_tile_dir) / f"{wsi_name}.class{lab_dict[k]}.object{j}.tile.jpg"
                    save_anno_path = pathlib.Path(save_tile_dir) / f"{wsi_name}.class{lab_dict[k]}.object{j}.annotation.json"
                    logging.info(f"saving tile {wsi_name}.class{lab_dict[k]}.object{j}.tile.jpg")
                    imwrite(save_tile_path, tile)
                    logging.info(f"saving annotation {wsi_name}.class{lab_dict[k]}.object{j}.annotation.json")
                    with save_anno_path.open("w") as handle:
                        json.dump(ann_dict, handle) 
        
    return

def construct_wsi_graph(wsi_paths, wsi_msk_paths, feature_mode, save_dir, mode, resolution=0.25, units="mpp"):
    """construct graph for each tile"""
    if feature_mode == "composition":
        _ = extract_composition_features(
            wsi_paths,
            wsi_msk_paths,
            save_dir,
            mode,
            resolution,
            units,
        )
    elif feature_mode == "cnn":
        _ = extract_deep_features(
            wsi_paths,
            wsi_msk_paths,
            save_dir,
            mode,
            resolution,
            units
        )
    else:
        raise ValueError(f"Invalid feature mode: {feature_mode}")
    
    for idx, wsi_path in enumerate(wsi_paths):
        wsi_name = pathlib.Path(wsi_path).stem
        graph_path = pathlib.Path(f"{save_dir}/{wsi_name}.json")
        logging.info("constructing graph: {}/{}...".format(idx + 1, len(wsi_paths)))
        construct_graph(wsi_name, save_dir, graph_path)
    return 

def generate_label_from_annotation(
        wsi_path,
        wsi_ann_path, 
        wsi_graph_path, 
        lab_dict, 
        ann_mpp, 
        min_ann_ratio, 
        resolution=0.25, 
        units="mpp"
    ):
    """
    Args:
        ann_mpp (float): given resolution of generating annotation, larger is cheaper
    """
    _, msk_readers, _ = prepare_annotation_reader(wsi_path, wsi_ann_path, lab_dict, ann_mpp, resolution, units)

    def _bbox_to_label(bbox):
        # bbox at given ann resolution
        bbox = bbox * resolution / ann_mpp
        bbox = bbox.astype(np.int32).tolist()
        if len(msk_readers) > 0:
            for k, msk_reader in msk_readers.items():
                bbox_region = msk_reader.read_bounds(bbox, ann_mpp, units, coord_space="resolution")
                if np.sum(bbox_region / 255.) / np.prod(bbox_region.shape) > min_ann_ratio:
                    label = lab_dict[k]
                    break
                else:
                    label = lab_dict["Background"]
        else:
            label = lab_dict["Background"]
        return label

    # generate label in parallel
    graph_dict = load_json(wsi_graph_path)
    xy = np.array(graph_dict["coordinates"])
    node_bboxes = np.concatenate((xy, xy + 512), axis=1)
    labels = joblib.Parallel(n_jobs=8)(
        joblib.delayed(_bbox_to_label)(bbox) for bbox in node_bboxes
        )
    return np.array(labels)

def generate_node_label(
        wsi_paths, 
        wsi_annot_paths, 
        lab_dict, 
        wsi_graph_paths, 
        save_lab_dir, 
        ann_mpp=4.0, 
        min_ann_ratio=1e-4, 
        resolution=0.25, 
        units="mpp"
    ):
    """generate node label for a list of wsi"""
    save_lab_dir = pathlib.Path(save_lab_dir)
    mkdir(save_lab_dir)
    for wsi_path, annot_path, graph_path in zip(wsi_paths, wsi_annot_paths, wsi_graph_paths):
        wsi_name = pathlib.Path(wsi_path).stem
        patch_labels = generate_label_from_annotation(wsi_path, annot_path, graph_path, lab_dict, ann_mpp, min_ann_ratio, resolution, units)
        save_lab_path = f"{save_lab_dir}/{wsi_name}.label.npy"
        logging.info(f"Saving node label {wsi_name}.label.npy")
        np.save(save_lab_path, patch_labels)
    return

def generate_data_split(
        x: list,
        y: list,
        train: float,
        valid: float,
        test: float,
        num_folds: int,
        seed: int = 5,
):
    """Helper to generate stratified splits
    """
    assert train + valid + test - 1.0 < 1.0e-10, "Ratios must sum to 1.0"

    outer_splitter = StratifiedShuffleSplit(
        n_splits=num_folds,
        train_size=train + valid,
        random_state=seed,
    )
    inner_splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train / (train + valid),
        random_state=seed,
    )

    splits = []
    for train_valid_idx, test_idx in outer_splitter.split(x, y):
        test_x = x[test_idx]
        test_y = y[test_idx]
        x_ = x[train_valid_idx]
        y_ = y[train_valid_idx]

        train_idx, valid_idx = next(iter(inner_splitter.split(x_, y_)))
        valid_x = x_[valid_idx]
        valid_y = y_[valid_idx]
        train_x = x_[train_idx]
        train_y = y_[train_idx]

        assert len(set(train_x).intersection(set(valid_x))) == 0
        assert len(set(valid_x).intersection(set(test_x))) == 0
        assert len(set(train_x).intersection(set(test_x))) == 0

        splits.append(
            {
                "train": list(zip(train_x, train_y)),
                "valid": list(zip(valid_x, valid_y)),
                "test": list(zip(test_x, test_y)),
            }
        )
    return splits

def create_pbar(subset_name: str, num_steps: int):
    """Create a nice progress bar."""
    pbar_format = (
        "Processing: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
    )
    pbar = tqdm(total=num_steps, leave=True, bar_format=pbar_format, ascii=True)
    if subset_name == "train":
        pbar_format += "step={postfix[1][step]:0.5f}|EMA={postfix[1][EMA]:0.5f}"
        # * Changing print char may break the bar so avoid it
        pbar = tqdm(
            total=num_steps,
            leave=True,
            initial=0,
            bar_format=pbar_format,
            ascii=True,
            postfix=["", {"step": float("NaN"), "EMA": float("NaN")}],
        )
    return pbar

def run_once(
        dataset_dict,
        num_epochs,
        save_dir,
        on_gpu=True,
        preproc_func=None,
        pretrained=None,
        loader_kwargs=None,
        arch_kwargs=None,
        optim_kwargs=None,
):
    """running the inference or training loop once"""
    if loader_kwargs is None:
        loader_kwargs = {}

    if arch_kwargs is None:
        arch_kwargs = {}

    if optim_kwargs is None:
        optim_kwargs = {}

    model = SlideGraphArch(**arch_kwargs)
    if pretrained is not None:
        model.load(*pretrained)
    model = model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), **optim_kwargs)

    loader_dict = {}
    for subset_name, subset in dataset_dict.items():
        _loader_kwargs = copy.deepcopy(loader_kwargs)
        ds = SlideGraphDataset(subset, mode=subset_name, preproc=preproc_func)
        loader_dict[subset_name] = DataLoader(
            ds,
            drop_last=subset_name == "train",
            shuffle=subset_name == "train",
            **_loader_kwargs,
        )
    
    for epoch in range(num_epochs):
        logger.info("EPOCH: %03d", epoch)
        for loader_name, loader in loader_dict.items():
            step_output = []
            ema = ScalarMovingAverage()
            pbar = create_pbar(loader_name, len(loader))
            for step, batch_data in enumerate(loader):
                if loader_name == "train":
                    output = model.train_batch(model, batch_data, on_gpu, optimizer)
                    ema({"loss": output[0]})
                    pbar.postfix[1]["step"] = step
                    pbar.postfix[1]["EMA"] = ema.tracking_dict["loss"]
                else:
                    output = model.infer_batch(model, batch_data, on_gpu)
                    batch_size = batch_data["graph"].num_graphs
                    output = [np.split(v, batch_size, axis=0) for v in output]
                    output = list(zip(*output))
                    step_output.append(output)
                pbar.update()
            pbar.close()

            logging_dict = {}
            if loader_name == "train":
                for val_name, val in ema.tracking_dict.items():
                    logging_dict[f"train-EMA-{val_name}"] = val
            elif "infer" in loader_name and any(v in loader_name for v in ["train", "valid"]):
                output = list(zip(*step_output))
                logit, true = output
                logit = np.squeeze(np.array(logit))
                true = np.squeeze(np.array(true))

                if "train" in loader_name:
                    scaler = PlattScaling()
                    scaler.fit(np.array(logit, ndmin=2).T, true)
                    model.aux_model["scaler"] = scaler
                scaler = model.aux_model["scaler"]
                prob = scaler.predict_proba(np.array(logit, ndmin=2).T)[:, 0]

                val = auroc_scorer(true, prob)
                logging_dict[f"{loader_name}-auroc"] = val
                val = auprc_scorer(true, prob)
                logging_dict[f"{loader_name}-auprc"] = val

                logging_dict[f"{loader_name}-raw-logit"] = logit
                logging_dict[f"{loader_name}-raw-true"] = true

            for val_name, val in logging_dict.items():
                if "raw" not in val_name:
                    logging.info("%s: %d", val_name, val)
            
            if "train" not in loader_dict:
                continue

            new_stats = {}
            if (save_dir / "stats.json").exists():
                old_stats = load_json(f"{save_dir}/stats.json")
                save_as_json(old_stats, f"{save_dir}/stats.old.json", exist_ok=False)
                new_stats = copy.deepcopy(old_stats)
                new_stats = {int(k): v for k, v in new_stats.items()}

            old_epoch_stats = {}
            if epoch in new_stats:
                old_epoch_stats = new_stats[epoch]
            old_epoch_stats.update(logging_dict)
            new_stats[epoch] = old_epoch_stats
            save_as_json(new_stats, f"{save_dir}/stats.json", exist_ok=False)

            model.save(
                f"{save_dir}/epoch={epoch:03d}.weights.pth",
                f"{save_dir}/epoch={epoch:03d}.aux.dat",
            )
    
    return step_output

def reset_logging(save_path):
    """Reset logger handler."""
    log_formatter = logging.Formatter(
        "|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
    )
    log = logging.getLogger()  # Root logger
    for hdlr in log.handlers[:]:  # Remove all old handlers
        log.removeHandler(hdlr)
    new_hdlr_list = [
        logging.FileHandler(f"{save_path}/debug.log"),
        logging.StreamHandler(),
    ]
    for hdlr in new_hdlr_list:
        hdlr.setFormatter(log_formatter)
        log.addHandler(hdlr)

def training(
        num_epochs,
        split_path,
        scaler_path,
        num_node_features,
        model_dir
):
    splits = joblib.load(split_path)
    node_scaler = joblib.load(scaler_path)
    loader_kwargs = {
        "num_workers": 8, 
        "batch_size": 16,
    }
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_target": 4,
        "layers": [16, 16, 8],
        "dropout": 0.5,
        "pooling": "mean",
        "conv": "EdgeConv",
        "aggr": "max",
    }
    optim_kwargs = {
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
    }
    if not pathlib.Path(model_dir).exists():
        for split_idx, split in enumerate(splits):
            new_split = {
                "train": split["train"],
                "infer-train": split["train"],
                "infer-valid-A": split["valid"],
                "infer-valid-B": split["test"],
            }
            split_save_dir = f"{model_dir}/{split_idx:02d}/"
            rm_n_mkdir(split_save_dir)
            reset_logging(split_save_dir)
            run_once(
                new_split,
                num_epochs,
                save_dir=split_save_dir,
                arch_kwargs=arch_kwargs,
                loader_kwargs=loader_kwargs,
                optim_kwargs=optim_kwargs,
                preproc_func=node_scaler.transform,
            )
    return

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
    chkpt_stats_list = stats[:top_k]  # select top_k

    model_dir = Path(stat_file_path).parent
    epochs = [v[0] for v in chkpt_stats_list]
    paths = [
        (
            f"{model_dir}/epoch={epoch:03d}.weights.pth",
            f"{model_dir}/epoch={epoch:03d}.aux.dat",
        )
        for epoch in epochs
    ]
    chkpt_stats_list = [[v[0], v[2]] for v in chkpt_stats_list]
    print(paths)  # noqa: T201
    return paths, chkpt_stats_list

def inference(
        split_path,
        scaler_path,
        num_node_features,
        pretrained_dir
):
    splits = joblib.load(split_path)
    node_scaler = joblib.load(scaler_path)
    loader_kwargs = {
        "num_workers": 8,
        "batch_size": 16,
    }
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_target": 1,
        "layers": [16, 16, 8],
        "dropout": 0.5,
        "pooling": "mean",
        "conv": "EdgeConv",
        "aggr": "max",
    }

    cum_stats = []
    for split_idx, split in enumerate(splits):
        new_split = {"infer": [v[0] for v in split["test"]]}

        stat_files = recur_find_ext(f"{pretrained_dir}/{split_idx:02d}/", [".json"])
        stat_files = [v for v in stat_files if ".old.json" not in v]
        assert len(stat_files) == 1
        chkpts, chkpt_stats_list = select_checkpoints(
            stat_files[0],
            top_k=1,
            metric="infer-valid-A",
        )

        # Perform ensembling by averaging probabilities
        # across checkpoint predictions
        cum_results = []
        for chkpt_info in chkpts:
            chkpt_results = run_once(
                new_split,
                num_epochs=1,
                save_dir=None,
                pretrained=chkpt_info,
                arch_kwargs=arch_kwargs,
                loader_kwargs=loader_kwargs,
                preproc_func=node_scaler.transform,
            )
            # * re-calibrate logit to probabilities
            model = SlideGraphArch(**arch_kwargs)
            model.load(*chkpt_info)
            scaler = model.aux_model["scaler"]
            chkpt_results = np.array(chkpt_results)
            chkpt_results = np.squeeze(chkpt_results)
            chkpt_results = scaler.transform(chkpt_results)

            cum_results.append(chkpt_results)
        cum_results = np.array(cum_results)
        cum_results = np.squeeze(cum_results)

        prob = cum_results
        if len(cum_results.shape) == 2:
            prob = np.mean(cum_results, axis=0)

        # * Calculate split statistics
        true = [v[1] for v in split["test"]]
        true = np.array(true)

        cum_stats.append(
            {"auroc": auroc_scorer(true, prob), "auprc": auprc_scorer(true, prob)},
        )
        return
    

def select_wsi_annotated(wsi_dir: str, ann_dir: str):
    """select annotated wsi
    Args:
        wsi_dir (str): directory of wsi
        ann_dir (str): directory of annotation
    Returns:
        selected_wsi_paths (list[pathlib.Path]): a list of selected wsi paths
        selected_ann_paths (list[pathlib.Path]): a list of selected annotation paths
    """
    
    ann_paths = sorted(pathlib.Path(ann_dir).rglob("*.json"))
    wsi_paths = sorted(pathlib.Path(wsi_dir).rglob("*HE.isyntax"))
    def _match_ann_wsi(ann_path, wsi_path):
        ann_name = ann_path.stem
        wsi_name = wsi_path.stem
        selected_paths = None
        if ann_name == wsi_name:
            selected_paths = [ann_path, wsi_path]
        return selected_paths

    selected_ann_wsi_paths = joblib.Parallel(n_jobs=8)(
        joblib.delayed(_match_ann_wsi)(ann_path, wsi_path) 
        for ann_path in ann_paths 
        for wsi_path in wsi_paths
        )

    selected_ann_paths = []
    selected_wsi_paths = []
    for paths in selected_ann_wsi_paths:
        if paths is not None:
            selected_ann_paths.append(paths[0])
            selected_wsi_paths.append(paths[1])
    return sorted(selected_wsi_paths), sorted(selected_ann_paths)   


if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', default="/well/rittscher/shared/datasets/KiBla/cases")
    parser.add_argument('--wsi_ann_dir', default="a_06semantic_segmentation/wsi_bladder_annotations")
    parser.add_argument('--save_dir', default="a_06semantic_segmentation", type=str)
    parser.add_argument('--mask_method', default='otsu', choices=["otsu", "morphological"], help='method of tissue masking')
    parser.add_argument('--task', default="bladder", choices=["bladder", "kidney"], type=str)
    parser.add_argument('--mode', default="tile", choices=["tile", "wsi"], type=str)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--feature_mode', default="composition", choices=["cnn", "composition"], type=str)
    parser.add_argument('--node_features', default=6, choices=[2048, 6], type=int)
    parser.add_argument('--resolution', default=0.25, type=float)
    parser.add_argument('--units', default="mpp", type=str)
    args = parser.parse_args()

    ## select annotated wsi
    wsi_dir = pathlib.Path(args.wsi_dir)
    wsi_ann_dir = pathlib.Path(args.wsi_ann_dir)
    wsi_paths, wsi_ann_paths = select_wsi_annotated(wsi_dir, wsi_ann_dir)
    logging.info("Totally {} wsi and {} annotation!".format(len(wsi_paths), len(wsi_ann_paths)))
    
    ## set save dir
    save_tile_dir = pathlib.Path(f"{args.save_dir}/wsi_{args.task}_tiles")
    save_msk_dir = pathlib.Path(f"{args.save_dir}/{args.mode}_{args.task}_masks")
    save_feature_dir = pathlib.Path(f"{args.save_dir}/{args.mode}_{args.task}_features")
    save_label_dir = pathlib.Path(f"{args.save_dir}/{args.mode}_{args.task}_labels")
    save_model_dir = pathlib.Path(f"{args.save_dir}/{args.mode}_{args.task}_models")
    

    ## define label name and value, should be consistent with annotation
    lab_dict = {
        "Background": 0,
        "Typical low grade urothelial carcinoma": 1,
        "Borderline/indeterminate for low grade vs high grade": 2,
        "Typical high grade urothelial carcinoma": 3,
    }
    ## set annotation resolution
    ann_mpp = 4.0 if args.mode == "wsi" else args.resolution

    ## generate ROI tile from wsi based on annotation
    if args.mode == "tile":
        # generate_tile_from_wsi(
        #     wsi_paths=wsi_paths,
        #     wsi_ann_paths=wsi_ann_paths,
        #     lab_dict=lab_dict,
        #     save_tile_dir=save_tile_dir,
        #     ann_mpp=ann_mpp,
        #     resolution=args.resolution,
        #     units=args.units,
        # )
        wsi_paths = sorted(save_tile_dir.glob("*.tile.jpg"))
        wsi_ann_paths = sorted(save_tile_dir.glob("*.annotation.json"))

    ## generate wsi tissue mask
    # if args.mode == "wsi":
    #     generate_wsi_tissue_mask(wsi_paths, save_msk_dir, args.mask_method)

    # ## construct wsi graph
    if args.mode == "wsi":
        save_msk_paths = sorted(save_msk_dir.glob("*.jpg"))
    else:
        save_msk_paths = None
    construct_wsi_graph(
        wsi_paths=wsi_paths,
        wsi_msk_paths=save_msk_paths,
        feature_mode=args.feature_mode,
        save_dir=save_feature_dir,
        mode=args.mode,
        resolution=args.resolution,
        units=args.units,
    )

    # ## generate node label from annotation
    # wsi_graph_paths = sorted(save_feature_dir.glob("*.json"))   
    # generate_node_label(
    #     wsi_paths=wsi_paths,
    #     wsi_annot_paths=wsi_ann_paths,
    #     lab_dict=lab_dict,
    #     wsi_graph_paths=wsi_graph_paths,
    #     save_lab_dir=save_label_dir,
    #     ann_mpp=ann_mpp,
    #     min_ann_ratio=1e-4,
    #     resolution=args.resolution,
    #     units=args.units,
    # )

    ## visualize a graph
    # wsi_path = pathlib.Path(wsi_paths[0])
    # graph_path = save_feature_dir / f"{wsi_path.stem}.json"
    # label_path = save_label_dir / f"{wsi_path.stem}.label.npy"
    # visualize_graph(wsi_path, graph_path, None, args.resolution, args.units)

    # ## split data set
    # num_folds = 5
    # test_ratio = 0.2
    # train_ratio = 0.8 * 0.9
    # valid_ratio = 0.8 * 0.1
    # wsi_graph_paths = sorted(save_feature_dir.glob("*.json"))
    # wsi_label_paths = sorted(save_label_dir.glob("*.label.npy"))
    # splits = generate_data_split(
    #     x=wsi_graph_paths,
    #     y=wsi_label_paths,
    #     train=train_ratio,
    #     valid=valid_ratio,
    #     test=test_ratio,
    #     num_folds=num_folds,
    # )
    # split_path = f"{save_model_dir}/splits.dat"
    # joblib.dump(splits, split_path)

    # ## compute mean and std on training data for normalization 
    # train_wsi_paths = [path for path, _ in splits[0]["train"]]
    # loader = SlideGraphDataset(train_wsi_paths, mode="infer")
    # loader = DataLoader(
    #     loader,
    #     num_workers=8,
    #     batch_size=1,
    #     shuffle=False,
    #     drop_last=False,
    # )
    # node_features = [v["graph"].x.numpy() for idx, v in enumerate(tqdm(loader))]
    # node_features = np.concatenate(node_features, axis=0)
    # node_scaler = StandardScaler(copy=False)
    # node_scaler.fit(node_features)
    # scaler_path = f"{save_model_dir}/node_scaler.dat"
    # joblib.dump(node_scaler, scaler_path)

    # ## training
    # training(
    #     num_epochs=args.epochs,
    #     split_path=split_path,
    #     scaler_path=scaler_path,
    #     num_node_features=args.node_features,
    #     model_dir=save_model_dir
    # )

    # ## inference
    # inference(
    #     split_path=split_path,
    #     scaler_path=scaler_path,
    #     num_node_features=args.node_features,
    #     pretrained_dir=save_model_dir
    # )




    



