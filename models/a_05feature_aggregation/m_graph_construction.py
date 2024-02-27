import sys
sys.path.append('../')

import os
import pathlib
import logging
import json
import torch
import joblib
import argparse
import numpy as np
import torch
import os
import pathlib
import json
import argparse
import logging

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from pprint import pprint
from common.m_utils import mkdir, load_json

from tiatoolbox.wsicore.wsireader import WSIReader, VirtualWSIReader
from tiatoolbox.tools.graph import SlideGraphConstructor
from tiatoolbox.utils.visualization import plot_graph, plot_map
from tiatoolbox.utils.misc import imwrite

from models.a_02tissue_masking.m_tissue_masking import generate_wsi_tissue_mask
from models.a_03patch_extraction.m_patch_extraction import prepare_annotation_reader
from models.a_04feature_extraction.m_feature_extraction import extract_cnn_features, extract_composition_features
torch.multiprocessing.set_sharing_strategy("file_system")

def construct_graph(wsi_name, wsi_feature_dir, save_path):
    positions = np.load(f"{wsi_feature_dir}/{wsi_name}.position.npy")
    features = np.load(f"{wsi_feature_dir}/{wsi_name}.features.npy")
    graph_dict = SlideGraphConstructor.build(positions[:, :2], features, feature_range_thresh=None)
    with save_path.open("w") as handle:
        new_graph_dict = {k: v.tolist() for k, v in graph_dict.items() if k != "cluster_points"}
        new_graph_dict.update({"cluster_points": graph_dict["cluster_points"]})
        json.dump(new_graph_dict, handle)

def construct_wsi_graph(wsi_paths, save_dir, n_jobs=8):
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
    joblib.Parallel(n_jobs=n_jobs)(
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
    wsi_thumb = wsi_reader.slide_thumbnail(resolution=16*resolution, units=units)
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

def visualize_graph(wsi_path, graph_path, label=None, positive_graph=False, show_map=False, resolution=0.25, units="mpp"):
    if pathlib.Path(wsi_path).suffix == ".jpg":
        NODE_RESOLUTION = {"resolution": resolution, "units": units}
        PLOT_RESOLUTION = {"resolution": 4*resolution, "units": units}
        NODE_SIZE = 24
        EDGE_SIZE = 4
    else:
        NODE_RESOLUTION = {"resolution": resolution, "units": units}
        PLOT_RESOLUTION = {"resolution": 16*resolution, "units": units}
        NODE_SIZE = 24
        EDGE_SIZE = 4
    graph_dict = load_json(graph_path)
    if show_map:
        cluster_points = graph_dict["cluster_points"]
        cluster_points = [np.array(c) for c in cluster_points]
        POINT_SIZE = [256, 256]
    graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items() if k != "cluster_points"}

    uncertainty_map = None
    if isinstance(label, pathlib.Path):
        node_activations = np.load(label)
    elif isinstance(label, np.ndarray):
        if label.ndim == 3:
            mean = np.mean(label, axis=0)
            std = np.std(label, axis=0)
            node_activations = np.argmax(mean, axis=1)
            uncertainty_map = np.array([std[i, node_activations[i]] for i in range(std.shape[0])])
        else:
            node_activations = np.argmax(label, axis=1)
    else:
        node_activations = np.argmax(graph_dict["x"].numpy(), axis=1)

    if positive_graph:
        positive = torch.tensor(node_activations).squeeze() > 0
        edge_index, _ = subgraph(positive, graph_dict["edge_index"], relabel_nodes=False)
        graph_dict["edge_index"] = edge_index
        
    graph_dict = {k: v.numpy() for k, v in graph_dict.items() if k != "cluster_points"}
    graph = Data(**graph_dict)

    cmap = get_cmap("viridis")
    norm_node_activations = (node_activations - node_activations.min()) / (node_activations.max() - node_activations.min() + 1e-10)
    node_colors = (cmap(norm_node_activations)[..., :3] * 255).astype(np.uint8)
    if uncertainty_map is not None:
        norm_uncertainty_map = (uncertainty_map - uncertainty_map.min()) / (uncertainty_map.max() - uncertainty_map.min() + 1e-10)
        uncertainty_colors = (cmap(norm_uncertainty_map)[..., :3] * 255).astype(np.uint8)

    # graph.x = StandardScaler().fit_transform(graph.x)
    # node_colors = PCA(n_components=3).fit_transform(graph.x)[:, [1,0,2]]
    # for channel in range(node_colors.shape[-1]):
    #     node_colors[:, channel] = 1 - equalize_hist(node_colors[:, channel]) ** 2
    # node_colors = (node_colors * 255).astype(np.uint8)

    if pathlib.Path(wsi_path).suffix == ".jpg":
        reader = WSIReader.open(wsi_path, mpp=(resolution, resolution))
    else:
        reader = WSIReader.open(wsi_path)

    node_resolution = reader.slide_dimensions(**NODE_RESOLUTION)
    plot_resolution = reader.slide_dimensions(**PLOT_RESOLUTION)
    fx = np.array(node_resolution) / np.array(plot_resolution)
    logging.info(f"The downsampling scale is {fx}")
    node_coordinates = np.array(graph.coordinates + 128) / fx
    edges = np.array(graph.edge_index.T)
    if show_map:
        cluster_points = [c / fx for c in cluster_points]
        POINT_SIZE = np.array(POINT_SIZE) / fx

    thumb = reader.slide_thumbnail(**PLOT_RESOLUTION)
    if uncertainty_map is not None:
        if not show_map:
            uncertainty_overlaid = plot_graph(
                thumb.copy(),
                node_coordinates,
                edges,
                node_colors=uncertainty_colors,
                node_size=NODE_SIZE,
                edge_size=EDGE_SIZE,
            )
            thumb_overlaid = plot_graph(
                thumb.copy(),
                node_coordinates,
                edges,
                node_colors=node_colors,
                node_size=NODE_SIZE,
                edge_size=EDGE_SIZE,
            )
        else:
            uncertainty_overlaid = plot_map(
                np.zeros_like(thumb),
                cluster_points,
                point_size=POINT_SIZE,
                cluster_colors=uncertainty_colors
            )
            thumb_overlaid = plot_map(
                thumb.copy(),
                cluster_points,
                point_size=POINT_SIZE,
                cluster_colors=node_colors
            )
        img_name = pathlib.Path(wsi_path).stem
        img_path = f"a_06semantic_segmentation/wsi_bladder_tumour_maps/{img_name}.png"
        imwrite(img_path, thumb_overlaid)
        plt.figure(figsize=(20,5))
        plt.subplot(1,2,1)
        plt.imshow(thumb)
        plt.axis("off")

        ax = plt.subplot(1,2,2)
        plt.imshow(thumb_overlaid)
        plt.axis("off")
        fig = plt.gcf()
        norm = Normalize(np.min(node_activations), np.max(node_activations))
        sm = ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, extend="both")
        cbar.minorticks_on()
        
        # ax = plt.subplot(1,2,2)
        # plt.imshow(uncertainty_overlaid)
        # plt.axis("off")
        # fig = plt.gcf()
        # norm = Normalize(np.min(uncertainty_map), np.max(uncertainty_map))
        # sm = ScalarMappable(cmap=cmap, norm=norm)
        # cbar = fig.colorbar(sm, ax=ax, extend="both")
        # cbar.minorticks_on()
        plt.savefig("a_05feature_aggregation/wsi_graph.jpg")
    else:
        if not show_map:
            thumb_overlaid = plot_graph(
                thumb.copy(),
                node_coordinates,
                edges,
                node_colors=node_colors,
                node_size=NODE_SIZE,
                edge_size=EDGE_SIZE,
            )
        else:
            thumb_overlaid = plot_map(
                thumb.copy(),
                cluster_points,
                point_size=POINT_SIZE,
                cluster_colors=node_colors
            )
        img_name = pathlib.Path(wsi_path).stem
        img_path = f"a_06semantic_segmentation/wsi_bladder_borderline/{img_name}.png"
        imwrite(img_path, thumb_overlaid)
        plt.figure(figsize=(20,5))
        plt.subplot(1,2,1)
        plt.imshow(thumb)
        plt.axis("off")
        ax = plt.subplot(1,2,2)
        plt.imshow(thumb_overlaid)
        plt.axis("off")
        fig = plt.gcf()
        norm = Normalize(np.min(node_activations), np.max(node_activations))
        sm = ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, extend="both")
        cbar.minorticks_on()
        plt.savefig("a_05feature_aggregation/wsi_graph.jpg")
    
def graph_feature_visualization(wsi_paths, save_graph_dir, save_label_dir, num_class, features=None, colors=None):
    if features is None or colors is None:
        features, colors = [], []
        for wsi_path in wsi_paths:
            wsi_name = pathlib.Path(wsi_path).stem
            logging.info(f"loading feature of {wsi_name}")
            graph_path = pathlib.Path(f"{save_graph_dir}/{wsi_name}.json")
            graph_dict = load_json(graph_path)
            feature = np.array(graph_dict["x"])
            features.append(feature)
            label_path = pathlib.Path(f"{save_label_dir}/{wsi_name}.label.npy")
            label = np.load(label_path)
            colors.append(label)
        features = np.concatenate(features, axis=0)
        colors = np.concatenate(colors, axis=0)
    
    pca_proj = PCA(n_components=64).fit_transform(features)
    tsne_proj = TSNE().fit_transform(pca_proj)

    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    palette = np.array(sns.color_palette("hls", num_class))
    plt.figure(figsize=(8,8))
    ax = plt.subplot(aspect='equal')
    class_list = np.unique(colors).tolist()
    scatter_list = []
    for i in class_list:
        c = colors[colors == i]
        t = tsne_proj[colors == i, :]
        sc = ax.scatter(t[:, 0], t[:, 1], lw=0, s=40, c=palette[c.astype(np.int32)])
        scatter_list.append(sc)
    plt.legend(scatter_list, class_list, loc="upper right", bbox_to_anchor=(1.1, 1.1), title="Classes")
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight') 
    plt.savefig('a_05feature_aggregation/feature_tsne_visualization.jpg')


if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', default="/well/rittscher/shared/datasets/KiBla/cases/3923_21/3923_21_G_HE.isyntax")
    parser.add_argument('--mask_method', default='morphological', help='method of tissue masking')
    parser.add_argument('--tile_location', default=[50000, 50000], type=list)
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument('--tile_size', default=[10240, 10240], type=list)
    parser.add_argument('--save_dir', default="a_05feature_aggregation/wsi_features", type=str)
    parser.add_argument('--mode', default="tile", type=str)
    parser.add_argument('--feature_mode', default="composition", type=str)
    parser.add_argument('--pre_generated', default=False, type=bool)
    parser.add_argument('--resolution', default=0.25, type=float)
    parser.add_argument('--units', default="mpp", type=str)
    args = parser.parse_args()

    if not args.pre_generated:
        wsi = WSIReader.open(args.slide_path)
        mask = wsi.tissue_mask(method=args.mask_method, resolution=1.25, units="power")
        pprint(wsi.info.as_dict())
    if args.mode == "tile":
        wsi_path = os.path.join("a_05feature_aggregation", 'tile_sample.jpg')
        msk_path = os.path.join("a_05feature_aggregation", 'tile_mask.jpg')
        if not args.pre_generated:
            tile = wsi.read_region(args.tile_location, args.level, args.tile_size)
            imwrite(wsi_path, tile)
            tile_mask = mask.read_region(args.tile_location, args.level, args.tile_size)
            imwrite(msk_path, np.uint8(tile_mask*255))  
    elif args.mode == "wsi":
        wsi_path = args.slide_path
        msk_path = os.path.join("a_05feature_aggregation", 'wsi_mask.jpg')
        if not args.pre_generated:
            wsi_mask = mask.slide_thumbnail(resolution=1.25, units="power")
            imwrite(msk_path, np.uint8(wsi_mask*255))
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    wsi_feature_dir = os.path.join(args.save_dir, args.feature_mode)
    if not args.pre_generated:
        wsi.close()
        if args.feature_mode == "composition":
            output_list = extract_composition_features(
                [wsi_path],
                [msk_path],
                wsi_feature_dir,
                args.mode,
                args.resolution,
                args.units,
            )
        elif args.feature_mode == "cnn":
            output_list = extract_cnn_features(
                [wsi_path],
                [msk_path],
                wsi_feature_dir,
                args.mode,
                args.resolution,
                args.units,
            )
        else:
            raise ValueError(f"Invalid feature mode: {args.feature_mode}")
    wsi_name = pathlib.Path(wsi_path).stem
    graph_path = pathlib.Path(f"{wsi_feature_dir}/{wsi_name}.json")
    if not args.pre_generated:
        construct_graph(wsi_name, wsi_feature_dir, graph_path)
    visualize_graph(wsi_path, graph_path, resolution=args.resolution, units=args.units)






