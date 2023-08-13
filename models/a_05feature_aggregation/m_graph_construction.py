import sys
sys.path.append('../')

import random
import torch
import os
import pathlib
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from torch_geometric.data import Data
from skimage.exposure import equalize_hist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pprint import pprint

from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.graph import SlideGraphConstructor
from tiatoolbox.utils.visualization import plot_graph
from tiatoolbox.utils.misc import imwrite

from models.a_04feature_extraction.m_feature_extraction import extract_deep_features, extract_composition_features
torch.multiprocessing.set_sharing_strategy("file_system")

SEED = 5
random.seed(SEED)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def load_json(path: str):
    with path.open() as fptr:
        return json.load(fptr)

def construct_graph(wsi_name, wsi_feature_dir, save_path):
    positions = np.load(f"{wsi_feature_dir}/{wsi_name}.position.npy")
    features = np.load(f"{wsi_feature_dir}/{wsi_name}.features.npy")
    graph_dict = SlideGraphConstructor.build(positions[:, :2], features, feature_range_thresh=None,)
    with save_path.open("w") as handle:
        graph_dict = {k: v.tolist() for k, v in graph_dict.items()}
        json.dump(graph_dict, handle)

def visualize_graph(wsi_path, graph_path, label_path=None, resolution=0.25, units="mpp"):
    NODE_RESOLUTION = {"resolution": resolution, "units": units}
    PLOT_RESOLUTION = {"resolution": 8*resolution, "units": units}
    NODE_SIZE = 24
    EDGE_SIZE = 4
    graph_dict = load_json(graph_path)
    graph_dict = {k: np.array(v) for k, v in graph_dict.items()}
    graph = Data(**graph_dict)

    if label_path is not None:
        node_activations = np.load(label_path)

    else:
        # node_activations = np.max(graph.x, axis=1)
        node_activations = np.std(graph.x, axis=1)

    cmap = get_cmap("viridis")
    norm_node_activations = (node_activations - node_activations.min()) / (node_activations.max() - node_activations.min() + 1e-10)
    node_colors = (cmap(norm_node_activations)[..., :3] * 255).astype(np.uint8)

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
    node_coordinates = np.array(graph.coordinates) / fx
    edges = np.array(graph.edge_index.T)

    thumb = reader.slide_thumbnail(**PLOT_RESOLUTION)
    thumb_overlaid = plot_graph(
        thumb.copy(),
        node_coordinates,
        edges,
        node_colors=node_colors,
        node_size=NODE_SIZE,
        edge_size=EDGE_SIZE,
    )
    plt.figure(figsize=(15,5))
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
    parser.add_argument('--feature_mode', default="cnn", type=str)
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
            output_list = extract_deep_features(
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






