import sys
sys.path.append('../')

import cv2
import os
import pathlib
import logging
import json
import copy
import torch
import joblib
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression as PlattScaling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score as auprc_scorer
from sklearn.metrics import roc_auc_score as auroc_scorer
from sklearn.metrics import accuracy_score as acc_scorer
from torch_geometric.loader import DataLoader

from tiatoolbox.utils.misc import save_as_json
from tiatoolbox.wsicore.wsireader import WSIReader, VirtualWSIReader, WSIMeta
from tiatoolbox.utils.misc import imwrite, imread
from tiatoolbox import logger
from tiatoolbox.tools import tissuemask

from models.a_04feature_extraction.m_feature_extraction import extract_composition_features
from models.a_04feature_extraction.m_feature_extraction import extract_deep_features
from models.a_05feature_aggregation.m_graph_construction import construct_graph
from models.a_05feature_aggregation.m_graph_construction import visualize_graph
from models.a_05feature_aggregation.m_graph_neural_network import SlideGraphArch
from models.a_05feature_aggregation.m_graph_neural_network import SlideGraphDataset
from models.a_05feature_aggregation.m_graph_neural_network import ScalarMovingAverage
from models.a_05feature_aggregation.m_graph_neural_network import PULoss

torch.multiprocessing.set_sharing_strategy("file_system")

import warnings
warnings.filterwarnings('ignore')

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
                full_path = f"{cur_path}/{file_name}"
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list

def load_json(path: str):
    path = pathlib.Path(path)
    with path.open() as fptr:
        return json.load(fptr)

def generate_wsi_tissue_mask(wsi_paths, save_msk_dir=None, method="otsu", resolution=1.25, units="power"):
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
                mpp=np.array([0.25, 0.25]),
                objective_power=40,
                axes="YXS",
                slide_dimensions=np.array(img.shape[:2][::-1]),
                level_downsamples=[1.0],
                level_dimensions=[np.array(img.shape[:2][::-1])],
                raw={"xml": None},
            )
            wsi = VirtualWSIReader(img, info=metadata)
        else:
            wsi = WSIReader.open(path)
        wsi_thumb = wsi.slide_thumbnail(resolution=resolution, units=units)
        return wsi_thumb
    
    # extract wsi thumbnails in parallel
    wsi_thumbs = joblib.Parallel(n_jobs=8)(
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

def prepare_annotation_reader(wsi_path, wsi_ann_path, lab_dict, resolution, units):
    """prepare annotation reader
    Args:
        wsi_path (str): path of a wsi of a tile
        wsi_ann_path (str): path of the annotation of the wsi or tile, should be in json format
        lab_dict (dict): a dict defines the label names and values, the label names should be the same
            as that in annotation file
        resolution (int): the resolution of preparing annotation
        units (str): the units of resolution, i.e., mpp
    Returns:
        wsi_reader (WSIReader): class::obj of reading wsi
        ann_reader (WSIReader): class::obj of reading annotation
        annotation (dict): loaded annotation file in dict format
    """
    assert units == "mpp", "units must be mpp"
    if pathlib.Path(wsi_path).suffix == ".jpg":
        img = imread(wsi_path)
        metadata = WSIMeta(
            mpp=np.array([0.25, 0.25]),
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
    wsi_shape = wsi_reader.slide_dimensions(resolution=resolution, units="mpp")
    msk_readers = {}
    for k in list(lab_dict.keys())[1:]:
        polygons = annotation[k]["points"]
        if len(polygons) > 0:
            img = np.zeros((wsi_shape[1], wsi_shape[0]), np.uint8)
            for polygon in polygons:
                # polygon at given ann resolution
                if len(polygon) > 0:
                    polygon = np.array(polygon) * wsi_reader.info.mpp / resolution
                    polygon = polygon.astype(np.int32)
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
                for j, bbox in enumerate(bounds):
                    if tile_size is not None:
                        bbox[0] = int((bbox[0] + bbox[2] - tile_size) / 2)
                        bbox[1] = int((bbox[1] + bbox[3] - tile_size) / 2)
                        bbox[2] = bbox[0] + tile_size
                        bbox[3] = bbox[1] + tile_size
                    tile = wsi_reader.read_bounds(bbox, resolution, units)
                    anno = ann_reader.read_bounds(bbox, resolution, units)
                    _, anno = cv2.threshold(anno, 0, 1, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(anno, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
    
    # extract tile in parallel
    joblib.Parallel(n_jobs=8)(
        joblib.delayed(_extract_tile_from_wsi)(idx, wsi_path, ann_path)
        for idx, (wsi_path, ann_path) in enumerate(zip(wsi_paths, wsi_ann_paths))
    )
        
    return

def extract_wsi_feature(
        wsi_paths, 
        wsi_msk_paths, 
        feature_mode, 
        save_dir, 
        mode, 
        resolution=0.25, 
        units="mpp"
    ):
    """extract feature from wsi
    Args:
        wsi_paths (list): a list of wsi paths
        wsi_msk_paths (list): a list of tissue mask paths of wsi
        fature_mode (str): mode of extracting features, 
            "composition" for extracting features by segmenting and counting nucleus
            "cnn" for extracting features by deep neural networks
        save_dir (str): directory of saving features
        mode (str): 'wsi' or 'tile', if 'wsi', extracting features of wsi
            could be slow if feature mode if 'composition'
        resolution (int): the resolution of extacting features
        units (str): the units of resolution, e.g., mpp  
    """
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
    return


def construct_wsi_graph(wsi_paths, save_dir):
    """construct graph for wsi
    Args:
        wsi_paths (list): a list of wsi paths
        save_dir (str): directory of reading feature and saving graph
    """
    def _construct_graph(idx, wsi_path):
        wsi_name = pathlib.Path(wsi_path).stem
        graph_path = pathlib.Path(f"{save_dir}/{wsi_name}.json")
        logging.info("constructing graph: {}/{}...".format(idx + 1, len(wsi_paths)))
        construct_graph(wsi_name, save_dir, graph_path)
        return
    
    # construct graphs in parallel
    joblib.Parallel(n_jobs=8)(
        joblib.delayed(_construct_graph)(idx, wsi_path)
        for idx, wsi_path in enumerate(wsi_paths)
    )    
    return 

def generate_label_from_annotation(
        wsi_path,
        wsi_ann_path, 
        wsi_graph_path, 
        tissue_masker,
        lab_dict, 
        node_size,
        min_ann_ratio, 
        resolution=0.25, 
        units="mpp"
    ):
    """generate label for each graph node according to annotation
    Args:
        wsi_path (str): path of a wsi
        wsi_ann_path (str): path of the annotation of given wsi
        wsi_graph_path (str): path of the graph of given wsi
        tissue_masker (Class): object of masking tissue
        lab_dict (dict): a dict defines the label names and values, the label names should be the same
            as that in annotation file
        node_size (int): the bounding box size for computing ratio of annotated area.
        min_ann_ratio (float): the required minimal annotation ratio of a node, should be 0 to 1
        resolution (int): the resolution of reading annotation, should be the same the resolution of node
        units (str): the units of resolution, e.g., mpp
    Returns:
        Array of labels of nodes
        Number of nodes
    """
    wsi_name = pathlib.Path(wsi_path).stem
    wsi_reader, ann_readers, _ = prepare_annotation_reader(
        wsi_path=wsi_path, 
        wsi_ann_path=wsi_ann_path, 
        lab_dict=lab_dict, 
        resolution=resolution, 
        units=units
    )
    wsi_thumb = wsi_reader.slide_thumbnail(resolution=resolution, units=units)
    msk_thumb = tissue_masker.transform([wsi_thumb])[0]
    msk_reader = VirtualWSIReader(msk_thumb.astype(np.uint8), info=wsi_reader.info, mode="bool")
    # imwrite(f"{wsi_name}_tissue_mask.jpg".replace("/", ""), msk_thumb.astype(np.uint8)*255)

    def _bbox_to_label(bbox):
        # bbox at given ann resolution
        bbox = bbox.astype(np.int32).tolist()
        if len(ann_readers) > 0:
            for k, ann_reader in ann_readers.items():
                bbox_ann = ann_reader.read_bounds(bbox, resolution, units, coord_space="resolution")
                bbox_msk = msk_reader.read_bounds(bbox, resolution, units, coord_space="resolution")
                anno = bbox_ann * bbox_msk
                if np.sum(anno) / np.prod(anno.shape) > min_ann_ratio:
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
    ## node size should be consistent with the output patch size of local feature extraction
    ## e.g., 164 for 
    node_bboxes = np.concatenate((xy, xy + node_size), axis=1)
    labels = joblib.Parallel(n_jobs=8)(
        joblib.delayed(_bbox_to_label)(bbox) for bbox in node_bboxes
        )
    if np.sum(np.array(labels) > 0) == 0:
        logging.warning(f"The nodes of {wsi_name} are all background!")
    return np.array(labels), len(xy)

def generate_node_label(
        wsi_paths, 
        wsi_annot_paths, 
        wsi_graph_paths,
        lab_dict,  
        save_lab_dir, 
        node_size=164, 
        min_ann_ratio=1e-4,
        resolution=0.25, 
        units="mpp"
    ):
    """generate node label for a list of graphs
    Args:
        wsi_paths (list): a list of wsi paths
        wsi_annot_paths (list): a list of annotation paths
        lab_dict (dict): a dict defines the label names and values, the label names should be the same
            as that in annotation file
        save_lab_dir (str): directory of saving labels
        node_size (int): the bounding box size for computing ratio of annotated area.
        min_ann_ratio (float): the required minimal annotation ratio of a node, should be 0 to 1
        resolution (int): the resolution of reading annotation, should be the same the resolution of node
        units (str): the units of resolution, e.g., mpp
    """
    save_lab_dir = pathlib.Path(save_lab_dir)
    mkdir(save_lab_dir)
    # finding threshold of masking tissue from all is better than each
    tissue_masker = generate_wsi_tissue_mask(
        wsi_paths=wsi_paths, 
        method="otsu", 
        resolution=16*resolution, 
        units=units
    )
    count_nodes = 0
    for idx, (wsi_path, annot_path, graph_path) in enumerate(zip(wsi_paths, wsi_annot_paths, wsi_graph_paths)):
        logging.info("annotating nodes of graph: {}/{}...".format(idx + 1, len(wsi_graph_paths)))
        wsi_name = pathlib.Path(wsi_path).stem
        patch_labels, num_nodes = generate_label_from_annotation(
            wsi_path=wsi_path, 
            wsi_ann_path=annot_path, 
            wsi_graph_path=graph_path, 
            tissue_masker=tissue_masker,
            lab_dict=lab_dict, 
            node_size=node_size, 
            min_ann_ratio=min_ann_ratio, 
            resolution=resolution, 
            units=units
        )
        count_nodes += num_nodes
        save_lab_path = f"{save_lab_dir}/{wsi_name}.label.npy"
        logging.info(f"Saving node label {wsi_name}.label.npy")
        np.save(save_lab_path, patch_labels)
    logging.info(f"Totally {count_nodes} nodes in {len(wsi_graph_paths)} graphs!")
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
    """Helper to generate splits
    Args:
        x (list): a list of image paths
        y (list): a list of annotation paths
        train (float): ratio of training samples
        valid (float): ratio of validating samples
        test (float): ratio of testing samples
        num_folds (int): number of folds for cross-validation
        seed (int): random seed
    Returns:
        splits (list): a list of folds, each fold consists of train, valid, and test splits
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

    l = []
    for path in y:
        label = np.array(np.load(pathlib.Path(path)))
        label = label[label > 0]
        label = np.unique(label)
        if len(label) == 0:
            logging.warning(f"All node labels are zeros in {path}!")
            l.append(0)
        else:
            index = np.random.randint(0, len(label))
            l.append(label[index])
    l = np.array(l)

    splits = []
    for train_valid_idx, test_idx in outer_splitter.split(x, l):
        test_x = [x[idx] for idx in test_idx]
        test_y = [y[idx] for idx in test_idx]
        x_ = [x[idx] for idx in train_valid_idx]
        y_ = [y[idx] for idx in train_valid_idx]
        l_ = [l[idx] for idx in train_valid_idx]

        train_idx, valid_idx = next(iter(inner_splitter.split(x_, l_)))
        valid_x = [x_[idx] for idx in valid_idx]
        valid_y = [y_[idx] for idx in valid_idx]
        train_x = [x_[idx] for idx in train_idx]
        train_y = [y_[idx] for idx in train_idx]

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
        pbar_format += "step={postfix[1][step]:03d}|EMA={postfix[1][EMA]:0.5f}"
        # * Changing print char may break the bar so avoid it
        pbar = tqdm(
            total=num_steps,
            leave=True,
            initial=0,
            bar_format=pbar_format,
            ascii=True,
            postfix=["", {"step": int(999), "EMA": float("NaN")}],
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
        probs=None,
        tau=1
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
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, **optim_kwargs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    # loss = torch.nn.CrossEntropyLoss()
    loss = PULoss(prior=0.35, mode="PN")

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
                    output = model.train_batch(model, batch_data, on_gpu, loss, optimizer, probs, tau)
                    ema({"loss": output[0]})
                    pbar.postfix[1]["step"] = step
                    pbar.postfix[1]["EMA"] = ema.tracking_dict["loss"]
                else:
                    output = model.infer_batch(model, batch_data, on_gpu)
                    batch_size = output[0].shape[0]
                    output = [np.split(v, batch_size, axis=0) for v in output]
                    output = list(zip(*output))
                    step_output += output
                pbar.update()
            pbar.close()

            logging_dict = {}
            if loader_name == "train":
                for val_name, val in ema.tracking_dict.items():
                    logging_dict[f"train-EMA-{val_name}"] = val
            elif "infer" in loader_name and any(v in loader_name for v in ["train", "valid"]):
                output = list(zip(*step_output))
                logit, true = output
                logit = np.array(logit).squeeze()
                true = np.array(true).squeeze()
                if logit.ndim == 1:
                    label = np.zeros_like(true)
                    sigmoid = 1 / (1 + np.exp(-logit))
                    label[sigmoid > 0.5] = 1
                    val = acc_scorer(label[true > 0], true[true > 0])
                else:
                    label = np.argmax(logit, axis=1)
                    val = acc_scorer(label, true)
                logging_dict[f"{loader_name}-acc"] = val

                logit = logit.reshape(-1, 1) if logit.ndim == 1 else logit
                if "train" in loader_name:
                    if logit.shape[1] == 1:
                        scaler = PlattScaling()
                    else:
                        scaler = PlattScaling(solver="saga", multi_class="multinomial")
                    scaler.fit(logit, true)
                    model.aux_model["scaler"] = scaler
                scaler = model.aux_model["scaler"]
                prob = scaler.predict_proba(logit)
                prob = prob[:, 1] if logit.shape[1] == 1 else prob
                val = auroc_scorer(true, prob, multi_class="ovr")
                logging_dict[f"{loader_name}-auroc"] = val

                if logit.shape[1] == 1:
                    val = auprc_scorer(true, prob)
                else:
                    onehot = np.eye(logit.shape[1])[true]
                    val = auprc_scorer(onehot, prob)
                logging_dict[f"{loader_name}-auprc"] = val

                logging_dict[f"{loader_name}-raw-logit"] = logit
                logging_dict[f"{loader_name}-raw-true"] = true

            for val_name, val in logging_dict.items():
                if "raw" not in val_name:
                    logging.info("%s: %0.5f\n", val_name, val)
            
            if "train" not in loader_dict:
                continue

            if (epoch + 1) % 10 == 0:
                new_stats = {}
                if (save_dir / "stats.json").exists():
                    old_stats = load_json(f"{save_dir}/stats.json")
                    save_as_json(old_stats, f"{save_dir}/stats.old.json", exist_ok=True)
                    new_stats = copy.deepcopy(old_stats)
                    new_stats = {int(k): v for k, v in new_stats.items()}

                old_epoch_stats = {}
                if epoch in new_stats:
                    old_epoch_stats = new_stats[epoch]
                old_epoch_stats.update(logging_dict)
                new_stats[epoch] = old_epoch_stats
                save_as_json(new_stats, f"{save_dir}/stats.json", exist_ok=True)
                model.save(
                    f"{save_dir}/epoch={epoch:03d}.weights.pth",
                    f"{save_dir}/epoch={epoch:03d}.aux.dat",
                )
        lr_scheduler.step()
    
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
        model_dir,
        probs=None,
        tau=1
):
    """train node classification neural networks
    Args:
        num_epochs (int): the number of epochs for training
        split_path (str): the path of storing data splits
        scaler_path (str): the path of storing data normalization
        num_node_features (int): the dimension of node feature
        model_dir (str): directory of saving models
    """
    splits = joblib.load(split_path)
    node_scaler = joblib.load(scaler_path)
    
    loader_kwargs = {
        "num_workers": 8, 
        "batch_size": 8,
    }
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_target": 1,
        "layers": [16, 16, 8], # [16, 16, 8]
        "dropout": 0.5,  #0.5
        "conv": "GINConv",
    }
    model_dir = model_dir / "GIN_PN"
    optim_kwargs = {
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,  # 1.0e-4
    }
    for split_idx, split in enumerate(splits):
        new_split = {
            "train": split["train"],
            "infer-train": split["train"],
            "infer-valid-A": split["valid"],
            "infer-valid-B": split["test"],
        }
        split_save_dir = pathlib.Path(f"{model_dir}/{split_idx:02d}/")
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
            probs=probs,
            tau=tau
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
    """node classification 
    """
    splits = joblib.load(split_path)
    node_scaler = joblib.load(scaler_path)
    
    loader_kwargs = {
        "num_workers": 8,
        "batch_size": 8,
    }
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_target": 4,
        "layers": [16, 16, 8],
        "dropout": 0.5,
        "conv": "GINConv"
    }
    pretrained_dir = pretrained_dir / "GIN"
    cum_stats = []
    for split_idx, split in enumerate(splits):
        new_split = {"infer": [v[0] for v in split["test"]]}

        stat_files = recur_find_ext(f"{pretrained_dir}/{split_idx:02d}/", [".json"])
        stat_files = [v for v in stat_files if ".old.json" not in v]
        assert len(stat_files) == 1
        chkpts, _ = select_checkpoints(
            stat_files[0],
            top_k=1,
            metric="infer-valid-A-auroc",
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
            chkpt_results = list(zip(*chkpt_results))
            chkpt_results = np.array(chkpt_results).squeeze()
            chkpt_results = scaler.predict_proba(chkpt_results)

            cum_results.append(chkpt_results)
        cum_results = np.array(cum_results)
        cum_results = np.squeeze(cum_results)

        prob = cum_results
        if len(cum_results.shape) == 3:
            prob = np.mean(cum_results, axis=0)

        # * Calculate split statistics
        true_paths = [v[1] for v in split["test"]]
        true = [np.load(f"{path}") for path in true_paths]
        true = np.concatenate(true, axis=0)
        onehot = np.eye(prob.shape[1])[true]

        cum_stats.append(
            {
                "auroc": auroc_scorer(true, prob, average=None, multi_class="ovr"), 
                "auprc": auprc_scorer(onehot, prob, average=None),
            }
        )
        print(f"Fold-{split_idx}:", cum_stats[-1])
    auroc_list = [stat["auroc"] for stat in cum_stats]
    auprc_list = [stat["auprc"] for stat in cum_stats]
    avg_stat = {
        "auroc": np.stack(auroc_list, axis=0).mean(axis=0),
        "auprc": np.stack(auprc_list, axis=0).mean(axis=0)
    }
    print(f"Avg:", avg_stat)
    return cum_stats

def test(
        graph_path,
        label_path,
        scaler_path,
        num_node_features,
        pretrained,
        conv="MLP"
):
    """node classification 
    """
    node_scaler = joblib.load(scaler_path)
    
    loader_kwargs = {
        "num_workers": 1,
        "batch_size": 1,
    }
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_target": 4,
        "layers": [16, 16, 8],
        "dropout": 0.5,
        "conv": conv
    }
    
    new_split = {"infer": [graph_path]}
    outputs = run_once(
        new_split,
        num_epochs=1,
        save_dir=None,
        pretrained=pretrained,
        arch_kwargs=arch_kwargs,
        loader_kwargs=loader_kwargs,
        preproc_func=node_scaler.transform,
    )

    # * re-calibrate logit to probabilities
    model = SlideGraphArch(**arch_kwargs)
    model.load(*pretrained)
    scaler = model.aux_model["scaler"]
    outputs = list(zip(*outputs))
    outputs = np.array(outputs).squeeze()
    prob = scaler.predict_proba(outputs)
    # true = np.load(f"{label_path}").astype(np.uint32)
    # true = np.array(true, np.int32).squeeze()
    # binary = np.zeros_like(true)
    # binary[true > 0] = 1
    # true_prob = np.zeros((len(true), 2), np.float32)
    # true_prob[binary] = prob[true]
    # auroc = auroc_scorer(true, true_prob, multi_class="ovr")
    # onehot = np.eye(2)[binary]
    # auprc = auprc_scorer(onehot, true_prob)
    # logging.info(f"AUROC: {auroc:0.5f}, AUPRC: {auprc:0.5f}")
    return prob
    

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


def compute_label_portion(split_path):
    splits = joblib.load(split_path)
    for idx, split in enumerate(splits):
        for name, subset in split.items():
            num_nodes = 0
            num_freqs = np.zeros([4], np.uint32)
            for _, path in subset:
                label = np.array(np.load(pathlib.Path(path)), np.uint32)
                num_nodes += len(label)
                uids, freqs = np.unique(label, return_counts=True)
                num_freqs[uids] = num_freqs[uids] + freqs
            p = (num_freqs / num_nodes).tolist()
            logging.info(f"Fold-{idx}: {name}: porition: [{p[0]:0.5f}, {p[1]:0.5f}, {p[2]:0.5f}, {p[3]:0.5f}]")
    return 




if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', default="/well/rittscher/shared/datasets/KiBla/cases")
    parser.add_argument('--wsi_ann_dir', default="a_06semantic_segmentation/wsi_bladder_annotations")
    parser.add_argument('--save_dir', default="a_06semantic_segmentation", type=str)
    parser.add_argument('--mask_method', default='otsu', choices=["otsu", "morphological"], help='method of tissue masking')
    parser.add_argument('--task', default="bladder", choices=["bladder", "kidney"], type=str)
    parser.add_argument('--mode', default="tile", choices=["tile", "wsi"], type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--feature_mode', default="cnn", choices=["cnn", "composition"], type=str)
    parser.add_argument('--node_features', default=2048, choices=[2048, 2], type=int)
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
    save_feature_dir = pathlib.Path(f"{args.save_dir}/{args.mode}_{args.task}_features/{args.feature_mode}")
    save_label_dir = pathlib.Path(f"{args.save_dir}/{args.mode}_{args.task}_labels/{args.feature_mode}")
    save_model_dir = pathlib.Path(f"{args.save_dir}/{args.mode}_{args.task}_models/{args.feature_mode}")
    

    ## define label name and value, should be consistent with annotation
    lab_dict = {
        "Background": 0,
        "Typical low grade urothelial carcinoma": 1,
        "Borderline/indeterminate for low grade vs high grade": 2,
        "Typical high grade urothelial carcinoma": 3,
    }
    ## set annotation resolution

    ## generate ROI tile from wsi based on annotation
    if args.mode == "tile":
        # generate_tile_from_wsi(
        #     wsi_paths=wsi_paths,
        #     wsi_ann_paths=wsi_ann_paths,
        #     lab_dict=lab_dict,
        #     save_tile_dir=save_tile_dir,
        #     tile_size=10240,
        #     resolution=args.resolution,
        #     units=args.units,
        # )
        wsi_paths = sorted(save_tile_dir.glob("*.tile.jpg"))
        wsi_ann_paths = sorted(save_tile_dir.glob("*.annotation.json"))
        logging.info("Totally {} tile and {} annotation!".format(len(wsi_paths), len(wsi_ann_paths)))

    ## generate wsi tissue mask
    # if args.mode == "wsi":
    #     generate_wsi_tissue_mask(wsi_paths, save_msk_dir, args.mask_method)

    ## extract wsi feature
    # if args.mode == "wsi":
    #     save_msk_paths = sorted(save_msk_dir.glob("*.jpg"))
    # else:
    #     save_msk_paths = None
    # extract_wsi_feature(
    #     wsi_paths=wsi_paths,
    #     wsi_msk_paths=save_msk_paths,
    #     feature_mode=args.feature_mode,
    #     save_dir=save_feature_dir,
    #     mode=args.mode,
    #     resolution=args.resolution,
    #     units=args.units,
    # )

    ## construct wsi graph
    # construct_wsi_graph(
    #     wsi_paths=wsi_paths,
    #     save_dir=save_feature_dir,
    # )

    # ## generate node label from annotation
    # wsi_graph_paths = sorted(save_feature_dir.glob("*.json")) 
    # node_size = 164 if args.feature_mode == "composition" else 224  
    # generate_node_label(
    #     wsi_paths=wsi_paths,
    #     wsi_annot_paths=wsi_ann_paths,
    #     wsi_graph_paths=wsi_graph_paths,
    #     lab_dict=lab_dict,
    #     save_lab_dir=save_label_dir,
    #     node_size=node_size,
    #     min_ann_ratio=0.1,
    #     resolution=args.resolution,
    #     units=args.units,
    # )

    ## split data set
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
    # num_train = len(splits[0]["train"])
    # logging.info(f"Number of training samples: {num_train}.")
    # num_valid = len(splits[0]["valid"])
    # logging.info(f"Number of validating samples: {num_valid}.")
    # num_test = len(splits[0]["test"])
    # logging.info(f"Number of testing samples: {num_test}.")
    # mkdir(save_model_dir)
    split_path = f"{save_model_dir}/splits.dat"
    # compute_label_portion(split_path)
    # joblib.dump(splits, split_path)

    ## compute mean and std on training data for normalization 
    # splits = joblib.load(split_path)
    # train_wsi_paths = [path for path, _ in splits[0]["train"]]
    # loader = SlideGraphDataset(train_wsi_paths, mode="infer")
    # loader = DataLoader(
    #     loader,
    #     num_workers=8,
    #     batch_size=1,
    #     shuffle=False,
    #     drop_last=False,
    # )
    # node_features = [v.x.numpy() for v in tqdm(loader)]
    # node_features = np.concatenate(node_features, axis=0)
    # node_scaler = StandardScaler(copy=False)
    # node_scaler.fit(node_features)
    scaler_path = f"{save_model_dir}/node_scaler.dat"
    # joblib.dump(node_scaler, scaler_path)

    ## training
    training(
        num_epochs=args.epochs,
        split_path=split_path,
        scaler_path=scaler_path,
        num_node_features=args.node_features,
        model_dir=save_model_dir,
        # probs=[0.65, 0.15, 0.05, 0.15],
        # tau=1
    )

    # ## inference
    # inference(
    #     split_path=split_path,
    #     scaler_path=scaler_path,
    #     num_node_features=args.node_features,
    #     pretrained_dir=save_model_dir
    # )

    ## visualize prediction
    # fold = 0
    # split = joblib.load(split_path)[fold]
    # graph_paths = [x for x, _ in split["test"]]
    # wsi_name = pathlib.Path(graph_paths[3]).stem
    # wsi_path = save_tile_dir / f"{wsi_name}.jpg"
    # graph_path = save_feature_dir / f"{wsi_name}.json"
    # label_path = save_label_dir / f"{wsi_name}.label.npy"
    # pretrained_model = f"a_06semantic_segmentation/tile_bladder_models/cnn/GIN/{fold:02d}/epoch=049.weights.pth"
    # pretrained_aux_model = f"a_06semantic_segmentation/tile_bladder_models/cnn/GIN/{fold:02d}/epoch=049.aux.dat"
    # prob = test(
    #     graph_path=graph_path,
    #     label_path=label_path,
    #     scaler_path=scaler_path,
    #     num_node_features=args.node_features,
    #     pretrained=[pretrained_model, pretrained_aux_model],
    #     conv="GINConv"
    # )
    # visualize_graph(wsi_path, graph_path, prob, args.resolution, args.units)





    



