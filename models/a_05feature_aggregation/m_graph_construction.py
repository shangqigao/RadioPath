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
import umap

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from scipy.special import softmax
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

import numpy as np
import networkx as nx
import plotly.graph_objects as go
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from torch_geometric import transforms
from pprint import pprint
from common.m_utils import mkdir, load_json, concat_dict_list

from tiatoolbox.wsicore.wsireader import WSIReader, VirtualWSIReader
from tiatoolbox.tools.graph import SlideGraphConstructor
from tiatoolbox.utils.visualization import plot_graph, plot_map
from tiatoolbox.utils.misc import imwrite

from models.a_02tissue_masking.m_tissue_masking import generate_wsi_tissue_mask
from models.a_03patch_extraction.m_patch_extraction import prepare_annotation_reader
from models.a_04feature_extraction.m_feature_extraction import extract_cnn_pathomic_features, extract_composition_features
torch.multiprocessing.set_sharing_strategy("file_system")

def construct_pathomic_graph(wsi_name, wsi_feature_dir, save_path):
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
        construct_pathomic_graph(wsi_name, save_dir, graph_path)
        return
    
    # construct graphs in parallel
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_construct_graph)(idx, wsi_path)
        for idx, wsi_path in enumerate(wsi_paths)
    )
    return 

def construct_radiomic_graph(
    img_name, 
    img_feature_dir, 
    save_path, 
    class_name="tumour", 
    patch_size=(30, 30, 30), 
    n_jobs=32
    ):
    """construct volumetric radiomic graph
    Args:
        patch_size (int): patch size for sliding window, reduce it if out of memory
    """
    positions = np.load(f"{img_feature_dir}/{img_name}_{class_name}_coordinates.npy")
    features = np.load(f"{img_feature_dir}/{img_name}_{class_name}_radiomics.npy")
    z, x, y = (np.max(positions, axis=0) - np.min(positions, axis=0) + 1).tolist()
    positions = np.reshape(positions, newshape=(z, x, y, -1))
    features = np.reshape(features, newshape=(z, x, y, -1))
    pz, px, py = patch_size

    start_Z, start_X, start_Y = np.mgrid[0:z:pz, 0:x:px, 0:y:py]
    start_Z, start_X, start_Y = start_Z.flatten(), start_X.flatten(), start_Y.flatten()
    end_Z, end_X, end_Y = start_Z + pz, start_X + px, start_Y + py
    logging.info(f"Splitting input volumetric feature into {len(start_X)} patches...")

    def _construct_graph(i):
        sz, sx, sy = start_Z[i], start_X[i], start_Y[i]
        ez, ex, ey = min(end_Z[i], z), min(end_X[i], x), min(end_Y[i], y)
        lz, lx, ly = ez - sz, ex - sx, ey - sy
        patch_positions = positions[sz:ez, sx:ex, sy:ey, :].reshape(lz*lx*ly, -1)
        patch_features = features[sz:ez, sx:ex, sy:ey, :].reshape(lz*lx*ly, -1)
        if lz*lx*ly < 4*pz*px:
            graph_dict = None
        else:
            graph_dict = SlideGraphConstructor.build(
                patch_positions, 
                patch_features, 
                lambda_d=0.2,
                lambda_f=0.1,
                lambda_h=0.8,
                connectivity_distance=8,
                neighbour_search_radius=4,
                feature_range_thresh=None
            )
        return {i: graph_dict}
    
    # construct graphs in parallel
    logging.info("Constructing graphs of patches in parallel...")
    list_graph_dicts = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_construct_graph)(i) for i in range(len(start_X))
    )

    # rearrange according to index
    all2one = {}
    for d in list_graph_dicts: all2one.update(d)
    list_graph_dicts = [all2one[i] for i in range(len(start_X)) if all2one[i] is not None]

    # concatenate a list of graphs
    logging.info("Concatenating all graphs constructed from the splitted patches...")
    new_graph_dict = {k: [] for k in list_graph_dicts[0].keys()}
    for graph_dict in list_graph_dicts:
        coordinate_shift = len(new_graph_dict["x"])
        for k, v in graph_dict.items():
            if k == "edge_index":
                new_graph_dict[k] += (v.T + coordinate_shift).tolist()
            elif k == "cluster_points":
                new_graph_dict[k] += v
            else:
                new_graph_dict[k] += v.tolist()
    num_nodes = len(new_graph_dict["x"])
    logging.info(f"Constructed a graph of {num_nodes} nodes from {z*x*y} features!")
    with save_path.open("w") as handle:
        json.dump(new_graph_dict, handle)

def construct_img_graph(img_paths, save_dir, class_name="tumour", patch_size=(30, 30, 30), n_jobs=32):
    """construct graph for radiological images
    Args:
        img_paths (list): a list of image paths
        save_dir (str): directory of reading feature and saving graph
    """
    for idx, img_path in enumerate(img_paths):
        img_name = pathlib.Path(img_path).name.replace(".nii.gz", "")
        graph_path = pathlib.Path(f"{save_dir}/{img_name}_{class_name}.json")
        logging.info("constructing graph: {}/{}...".format(idx + 1, len(img_paths)))
        construct_radiomic_graph(img_name, save_dir, graph_path, class_name, patch_size, n_jobs)
    return 


def extract_minimum_spanning_tree(wsi_graph_paths, save_dir, n_jobs=8):
    """extract minimum spanning tree from graph
    Args:
        wsi_graph_paths (list): a list of wsi graph paths
        save_dir (str): directory of saving minimum spanning tree
    """
    def _extract_mst(idx, graph_path):
        with pathlib.Path(graph_path).open() as fptr:
            original_graph_dict = json.load(fptr)
        graph_dict = {k: np.array(v) for k, v in original_graph_dict.items() if k != "cluster_points"}
        logging.info("extracting minimum spanning tree: {}/{}...".format(idx + 1, len(wsi_graph_paths)))
        feature = graph_dict["x"]
        edge_index = graph_dict["edge_index"]
        rows, cols = edge_index[0, :], edge_index[1, :]
        distance = np.linalg.norm(feature[rows] - feature[cols], axis=1)
        shape = (len(feature), len(feature))
        X = csr_matrix((distance, (rows, cols)), shape)
        MST = minimum_spanning_tree(X).toarray()
        edge_index = np.ascontiguousarray(np.stack(MST.nonzero(), axis=1).T)
        graph_dict.update({"edge_index": edge_index.astype(np.int64)})
        wsi_name = pathlib.Path(graph_path).stem
        save_path = pathlib.Path(f"{save_dir}/{wsi_name}.MST.json")
        with save_path.open("w") as handle:
            graph_dict = {k: v.tolist() for k, v in graph_dict.items()}
            graph_dict.update({"cluster_points": original_graph_dict["cluster_points"]})
            json.dump(graph_dict, handle)

    # extract minimum spanning trees in parallel
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_extract_mst)(idx, graph_path)
        for idx, graph_path in enumerate(wsi_graph_paths)
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
        resolution=0.5, 
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

def generate_label_from_classification(
        wsi_graph_path,
        wsi_cls_path,
        n_jobs
):
    with pathlib.Path(wsi_graph_path).open() as fptr:
        graph_dict = json.load(fptr)
    cluster_points_list = graph_dict["cluster_points"]

    patch_cls_np = np.load(wsi_cls_path)
    wsi_pos_path = f"{wsi_cls_path}".replace(".SimilarityScores.npy", ".position.npy")
    patch_pos_np = np.load(wsi_pos_path)[:, :2]

    def _label_graph_node(idx, cluster_points):
        cls_prob_list = []
        for point in cluster_points:
            point = np.array([point])
            index = np.argwhere(np.sum(patch_pos_np == point, axis=1) == 2).squeeze()
            cls_prob_list.append(patch_cls_np[index])
        label = np.stack(cls_prob_list, axis=0).mean(axis=0)
        return idx, label

    # construct graphs in parallel
    idx_labels = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_label_graph_node)(idx, cluster_points)
        for idx, cluster_points in enumerate(cluster_points_list)
    )
    labels = np.zeros((len(cluster_points_list), patch_cls_np.shape[1]))
    for idx, label in idx_labels: labels[idx] = label
    
    return labels, len(labels)

def generate_node_label(
        wsi_paths, 
        wsi_annot_paths, 
        wsi_graph_paths, 
        save_lab_dir,
        anno_type,
        lab_dict=None,
        n_jobs=8,
        node_size=164, 
        min_ann_ratio=1e-4,
        resolution=0.5, 
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
    if anno_type == "annotation":
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
        if anno_type == "annotation":
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
        elif anno_type == "classification":
            patch_labels, num_nodes = generate_label_from_classification(
                wsi_graph_path=graph_path,
                wsi_cls_path=annot_path,
                n_jobs=n_jobs
            )
        else:
            raise NotImplementedError
        count_nodes += num_nodes
        save_lab_path = f"{save_lab_dir}/{wsi_name}.label.npy"
        logging.info(f"Saving node label {wsi_name}.label.npy")
        np.save(save_lab_path, patch_labels)
    logging.info(f"Totally {count_nodes} nodes in {len(wsi_graph_paths)} graphs!")
    return

def measure_subgraph_properties(
    graph_path,
    label_path,
    subgraph_ids=None,    
    ):
    graph_dict = load_json(graph_path)
    
    if subgraph_ids is None:
        graph_dict = {k: np.array(v) for k, v in graph_dict.items() if k != "cluster_points"}
        feature = graph_dict["x"]
        edge_index = graph_dict["edge_index"]
    else:
        graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items() if k != "cluster_points"}
        label = np.load(label_path)
        if label.ndim == 2: label = np.argmax(label, axis=1)
        label = torch.tensor(label).squeeze()
        subset = torch.logical_and(label >= subgraph_ids[0], label <= subgraph_ids[1])
        if subset.sum().item() < 2: return None
        edge_index, _ = subgraph(subset, graph_dict["edge_index"], relabel_nodes=True)
        if edge_index.size(1) == 0: return None
        edge_index = edge_index.numpy()
        feature = graph_dict["x"][subset].numpy()
    
    rows, cols = edge_index[0, :], edge_index[1, :]
    distance = np.linalg.norm(feature[rows] - feature[cols], axis=1)
    shape = (len(feature), len(feature))
    X = csr_matrix((distance, (rows, cols)), shape)
    G = nx.Graph(X)

    prop_dict = {}
    prop_dict["num_nodes"] = [G.number_of_nodes()]
    prop_dict["num_edges"] = [G.number_of_edges()]
    prop_dict["num_components"] = [nx.number_connected_components(G)]
    prop_dict["degree"] = list(list(zip(*G.degree()))[1])
    prop_dict["closeness"] = list(nx.closeness_centrality(G, distance="weight").values())
    prop_dict["graph_diameter"] = [nx.diameter(G.subgraph(SG)) for SG in nx.connected_components(G)]
    prop_dict["graph_assortativity"] = [nx.degree_pearson_correlation_coefficient(G)]
    prop_dict["mean_neighbor_degree"] = list(nx.average_neighbor_degree(G).values())

    return prop_dict

def measure_graph_properties(
    graph_paths,
    label_paths,
    save_dir,
    subgraph_dict=None,
    n_jobs=8
    ):
    def _measure_graph_properties(idx, graph_path, label_path):
        logging.info("Measuring graph properties: {}/{}...".format(idx + 1, len(graph_paths)))
        wsi_name = pathlib.Path(graph_path).stem
        if subgraph_dict is None:
            prop_dict = measure_subgraph_properties(graph_path, label_path)
            save_path = pathlib.Path(f"{save_dir}/{wsi_name}.graph.properties.json")
        else:
            prop_dict = {}
            for cls, ids in subgraph_dict.items():
                subgraph_prop_dict = measure_subgraph_properties(graph_path, label_path, ids)
                prop_dict.update({cls: subgraph_prop_dict})
            save_path = pathlib.Path(f"{save_dir}/{wsi_name}.subgraphs.properties.json")
        with save_path.open("w") as handle:
            json.dump(prop_dict, handle)
    
    # measure graph properties in parallel
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_measure_graph_properties)(idx, graph_path, label_path)
        for idx, (graph_path, label_path) in enumerate(zip(graph_paths, label_paths))
    )
    return

def plot_graph_properties(
    prop_paths,
    subgraph_dict=None,
    prop_key="num_nodes",
    plotted="hist",
    min_percentile=0,
    max_percentile=100
    ):

    prop_list = [load_json(p) for p in prop_paths]
    
    if subgraph_dict is None:
        graph_prop_dict = concat_dict_list(prop_list)
    else:
        graph_prop_dict = {}
        for k in subgraph_dict.keys():
            dict_list = [d[k] for d in prop_list if d[k] is not None]
            graph_prop_dict[k] = concat_dict_list(dict_list)

    property_dict = {}
    if subgraph_dict is None:
        assert prop_key in graph_prop_dict.keys()
        property_dict.update({"Whole slide graph": graph_prop_dict[prop_key]})
    else:
        for k in subgraph_dict.keys():
            property_dict.update({f"Subgraph {k}": graph_prop_dict[k][prop_key]})

    # plot
    logging.info(f"Visualizing {plotted} of the graph property {prop_key}...")
    plt.figure()
    i, D = 1, []
    for k, v in property_dict.items():
        y = np.array(v)
        y = y[~np.isnan(y)]
        if prop_key == "graph_diameter": y = y[y > 1]
        min_perc = np.percentile(y, min_percentile)
        max_perc = np.percentile(y, max_percentile)
        y = y[(y >= min_perc) & (y <= max_perc)]
        D.append(y)
        if plotted in ["bar", "stem", "hist"]:
            ax = plt.subplot(1, len(property_dict), i)
            if plotted == "bar":
                x = np.arange(len(k)) + 0.5
                ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
                ax.set_xlabel("Subject ID")
                ax.set_ylabel(f"{prop_key}")
            elif plotted == "stem":
                x = np.arange(len(k)) + 0.5
                ax.stem(x, y, label=k)
                ax.set_xlabel("Subject ID")
                ax.set_ylabel(f"{prop_key}")
            elif plotted == "hist":
                ax.hist(y, bins="auto", linewidth=0.5, edgecolor="white")
                ax.set_xlabel(f"{prop_key}")
                ax.set_ylabel("Frequency")
            ax.set_title(f"{prop_key} for {k}")
        i += 1

    if plotted in ["box", "voilin", "plot"]:
        ax = plt.subplot(1,1,1)
        if plotted == "box":
            positions = [(i+1)*2 for i in range(len(graph_prop_dict))]
            ax.boxplot(D, positions=positions, widths=1.5, patch_artist=True,
                    showmeans=False, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5}
                    )
            ax.set_ylabel(f"{prop_key}")
            ax.set_xticks(positions)
            ax.set_xticklabels(property_dict.keys(), rotation=45, ha='right')
        elif plotted == "voilin":
            positions = [(i+1)*2 for i in range(len(graph_prop_dict))]
            ax.violinplot(D, positions, widths=2,
                    showmeans=False, showmedians=True, showextrema=True
                    )
            ax.set_ylabel(f"{prop_key}")
            ax.set_xticks(positions)
            ax.set_xticklabels(property_dict.keys(), rotation=45, ha='right')
        elif plotted == "plot":
            for d in D:
                x = np.arange(len(d))
                ax.plot(x, d, linewidth=2)
            ax.set_xlabel("subject ID")
            ax.set_ylabel(f"{prop_key}")
        ax.set_title(f"Comparison of {prop_key}")
    plt.savefig(f"a_05feature_aggregation/graph_{prop_key}.jpg") 
    logging.info("Visualization done!") 
    return

def visualize_pathomic_graph(
        wsi_path, 
        graph_path, 
        label=None, 
        label_min=None,
        label_max=None,
        subgraph_id=None, 
        show_map=False, 
        magnify=False, 
        resolution=0.5, 
        units="mpp",
        save_name="pathomics",
        save_title="pathomics graph",
        save_wsi=False
    ):
    if pathlib.Path(wsi_path).suffix == ".jpg":
        NODE_RESOLUTION = {"resolution": resolution, "units": units}
        PLOT_RESOLUTION = {"resolution": resolution / 4, "units": units}
        NODE_SIZE = 24
        EDGE_SIZE = 4
    else:
        NODE_RESOLUTION = {"resolution": resolution, "units": units}
        PLOT_RESOLUTION = {"resolution": resolution / 16, "units": units}
        NODE_SIZE = 12
        EDGE_SIZE = 2
    graph_dict = load_json(graph_path)
    if show_map:
        cluster_points = graph_dict["cluster_points"]
        cluster_points = [np.array(c) for c in cluster_points]
        POINT_SIZE = [256, 256] # patch size in embedding
    graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items() if k != "cluster_points"}

    uncertainty_map = None
    if isinstance(label, pathlib.Path) or isinstance(label, str):
        node_activations = np.load(label)
        if node_activations.ndim == 2:
            node_activations = np.argmax(node_activations, axis=1)
    elif isinstance(label, np.ndarray):
        if label.ndim == 3:
            mean = np.mean(label, axis=0)
            std = np.std(label, axis=0)
            node_activations = np.argmax(mean, axis=1)
            uncertainty_map = np.array([std[i, node_activations[i]] for i in range(std.shape[0])])
        elif label.ndim == 2:
            node_activations = np.argmax(label, axis=1)
        elif label.ndim == 1:
            node_activations = label
    else:
        node_activations = np.argmax(softmax(graph_dict["x"].numpy(), axis=1), axis=1)
    if label_min is None: label_min = node_activations.min()
    if label_max is None: label_max = node_activations.max()

    if subgraph_id is not None:
        act_tensor = torch.tensor(node_activations).squeeze()
        subset = torch.logical_and(act_tensor >= subgraph_id[0], act_tensor <= subgraph_id[1])
        edge_index, _ = subgraph(subset, graph_dict["edge_index"], relabel_nodes=True)
        subset = subset.numpy().tolist()
        node_activations = node_activations[subset]
        if show_map: cluster_points = cluster_points[subset]
        graph_dict = {k: v[subset] for k, v in graph_dict.items() if k != "edge_index"}
        graph_dict["edge_index"] = edge_index

    graph_dict = {k: v.numpy() for k, v in graph_dict.items()}
    graph = Data(**graph_dict)

    cmap = get_cmap("viridis")
    norm_node_activations = (node_activations - label_min) / (label_max - label_min + 1e-10)
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
    if save_wsi:
        wsi = reader.slide_thumbnail(resolution=10, units="power")
        wsi_path = "a_05feature_aggregation/wsi.jpg"
        imwrite(wsi_path, wsi)
        
    thumb_bbox = None
    if magnify:
        tile_node_coordinates = node_coordinates * fx / 4
        n_nodes = len(tile_node_coordinates)
        adjacent_matric = np.zeros((n_nodes, n_nodes))
        for i in range(len(edges)):
            x, y = edges[i][0], edges[i][1]
            adjacent_matric[x, y] = 1
        ul = np.mean(tile_node_coordinates, axis=0)
        x_start, y_start = ul[0] - 1300, ul[1] - 1300
        x_end, y_end = x_start + 2600, y_start + 2600
        tile_bbox = np.array([x_start, y_start, x_end, y_end])
        thumb_bbox = tile_bbox * 4 / np.concatenate([fx]*2)
        bbox_index = []
        for i in range(len(tile_node_coordinates)):
            x, y = tile_node_coordinates[i][0], tile_node_coordinates[i][1]
            if (x_start <= x < x_end) and (y_start <= y < y_end):
                bbox_index.append(i)
        tile_node_coordinates = tile_node_coordinates[bbox_index]
        tile_node_coordinates -= np.array([[x_start, y_start]])
        tile_node_colors = node_colors[bbox_index]
        tile_adjacent_matrix = adjacent_matric[bbox_index, :][:, bbox_index]
        n_nodes = len(tile_node_coordinates)
        tile_edges = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if tile_adjacent_matrix[i, j] == 1:
                    tile_edges.append([i, j])
        tile_edges = np.array(tile_edges)
        rect_params = {
            "location": (int(x_start), int(y_start)),
            "size": (int(x_end - x_start), int(y_end - y_start)),
            "resolution": NODE_RESOLUTION["resolution"] / 4, 
            "units": units,
            "coord_space": "resolution"
            }
        tile = reader.read_rect(**rect_params)

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
        norm = Normalize(label_min, label_max)
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
                bbox=thumb_bbox
            )
        else:
            thumb_overlaid = plot_map(
                thumb.copy(),
                cluster_points,
                point_size=POINT_SIZE,
                cluster_colors=node_colors
            )
        # img_name = pathlib.Path(wsi_path).stem
        # img_path = f"a_06generative_SR/{img_name}.png"
        # imwrite(img_path, thumb_overlaid)
        if magnify:
            thumb_tile = plot_graph(
                tile.copy(),
                tile_node_coordinates,
                tile_edges,
                node_colors=tile_node_colors,
                node_size=NODE_SIZE,
                edge_size=EDGE_SIZE,
            )
        else:
            thumb_tile = np.ones((256, 256, 3))*255
        plt.figure(figsize=(20,10))
        plt.subplot(2,2,1)
        plt.imshow(thumb)
        plt.axis("off")
        ax = plt.subplot(2,2,2)
        plt.imshow(thumb_overlaid)
        plt.title(f"{save_title}")
        plt.axis("off")
        fig = plt.gcf()
        norm = Normalize(label_min, label_max)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, extend="both")
        cbar.minorticks_on()
        plt.subplot(2,2,3)
        plt.imshow(thumb_tile)
        plt.savefig(f"a_05feature_aggregation/wsi_graph_{save_name}.jpg")

def visualize_radiomic_graph(
        img_path,
        lab_path, 
        save_graph_dir,
        attention=None,
        class_name="tumour",
        save_name="radiomics",
        keys=["image", "label"],
        spacing=(1.024, 1.024, 1.024),
        padding=(4, 8, 8),
        NODE_SIZE = 1,
        EDGE_SIZE = 0.25,
        remove_front_corner=True,
        n_jobs=32
    ):
    from models.a_04feature_extraction.m_feature_extraction import SegVol_image_transforms
    from monai.transforms.utils import generate_spatial_bounding_box
    from monai.transforms.utils import get_largest_connected_component_mask

    transform = SegVol_image_transforms(keys, spacing, padding)
    case_dict = [{"image": img_path, "label": lab_path}]
    data = transform(case_dict)
    image, label = data[0]["image"].squeeze(), data[0]["label"].squeeze()
    label = get_largest_connected_component_mask(label)
    s, e = generate_spatial_bounding_box(np.expand_dims(label, 0))
    padding = (4, 8, 8)
    s = np.array(s) - np.array(padding)
    e = np.array(e) + np.array(padding)
    print(s, e)

    img_name = pathlib.Path(img_path).name.replace(".nii.gz", "")
    logging.info(f"loading graph of {img_name}")
    graph_path = pathlib.Path(f"{save_graph_dir}/{img_name}_{class_name}.json")
    graph_dict = load_json(graph_path)
    cluster_points = graph_dict["cluster_points"]
    graph_dict = {k: v for k, v in graph_dict.items() if k != "cluster_points"}
    node_coordinates = graph_dict["coordinates"]
    print(f"The number of nodes: {len(node_coordinates)}")
    node_label = [label[int(c[0]), int(c[1]), int(c[2])] for c in node_coordinates]

    edges = graph_dict["edge_index"]
    c = (np.array(e) + np.array(s)) // 2
    def _condition(n, c): return any([n[0]<c[0], n[1]<c[1], n[2]<c[2]])
    kept_nodes = [_condition(n, c) for n in node_coordinates]
    edge_nodes = [(node_coordinates[p1], node_coordinates[p2]) for p1, p2 in edges]
    kept_edges = [all([_condition(s, c), _condition(e, c)]) for s, e in edge_nodes]
    logging.info(f"loaded graph !!!")
    
    if attention is None:
        voi = image[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
    else:
        def _assign_attention(points, value):
            voi = np.zeros((e[0]-s[0], e[1]-s[1], e[2]-s[2]))
            for p in points: 
                p[0] = p[0] - s[0]
                p[1] = p[1] - s[1]
                p[2] = p[2] - s[2]
                voi[p[0], p[1], p[2]] = value
            return voi
        # assign attention in parallel
        voi_list = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_assign_attention)(points, value)
            for points, value in zip(cluster_points, attention.tolist())
        )
        voi = sum(voi_list)
        voi /= voi.max()

    X, Y, Z = np.mgrid[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=voi.flatten(),
        isomin=0.1,
        isomax=0.8,
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=17, # needs to be a large number for good volume rendering
        ))
    fig.write_image(f"a_05feature_aggregation/image_voi_{save_name}.jpg")

    node_coordinates = np.array(node_coordinates) 
    kept_nodes = np.array(kept_nodes)
    if remove_front_corner:
        X = node_coordinates[kept_nodes, 0]
        Y = node_coordinates[kept_nodes, 1]
        Z = node_coordinates[kept_nodes, 2]
        node_color = np.array(node_label)[kept_nodes].tolist()
    else:
        X = node_coordinates[:, 0]
        Y = node_coordinates[:, 1]
        Z = node_coordinates[:, 2]
        node_color = node_label
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='markers',
        marker=dict(
            size=NODE_SIZE,
            color=node_color,  # Use scalar values to determine color
            colorscale='Viridis',  # Choose a colormap (e.g., 'Viridis', 'Cividis', 'Plasma', etc.)
            colorbar=dict(title='Label')  # Add a colorbar to show the mapping
        ),
        name='Points'
    ))
    
    x_coords = []
    y_coords = []
    z_coords = []
    edge_colors = []
    edges = np.array(edges)
    kept_edges = np.array(kept_edges)
    if remove_front_corner: edges = edges[kept_edges].tolist()
    for edge in edges:
        p1, p2 = edge
        edge_colors.append(node_label[p1])
        x_coords.extend([node_coordinates[p1, 0], node_coordinates[p2, 0], None])
        y_coords.extend([node_coordinates[p1, 1], node_coordinates[p2, 1], None])
        z_coords.extend([node_coordinates[p1, 2], node_coordinates[p2, 2], None])
    
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='lines',
        line=dict(color=edge_colors, colorscale='Viridis', width=EDGE_SIZE),
        name='Edges'
    ))

    fig.update_layout(
        title='Radiological graph of VOI',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.write_image(f"a_05feature_aggregation/image_graph_{save_name}.jpg")
    print("Visualization Done!")
    
def pathomic_feature_visualization(wsi_paths, save_feature_dir, mode="tsne", save_label_dir=None, graph=True, n_class=None, features=None, colors=None):
    if features is None or colors is None:
        features, colors = [], []
        for i, wsi_path in enumerate(wsi_paths):
            wsi_name = pathlib.Path(wsi_path).stem
            logging.info(f"loading feature of {wsi_name}")
            if graph:
                feature_path = pathlib.Path(f"{save_feature_dir}/{wsi_name}.json")
                graph_dict = load_json(feature_path)
                feature = np.array(graph_dict["x"])
            else:
                feature_path = pathlib.Path(f"{save_feature_dir}/{wsi_name}.features.npy")
                feature = np.load(feature_path)
            features.append(feature)
            if save_label_dir is not None:
                if graph:
                    label_path = pathlib.Path(f"{save_label_dir}/{wsi_name}.label.npy")
                else:
                    label_path = pathlib.Path(f"{save_label_dir}/{wsi_name}.features.npy")
                label = np.load(label_path)
                if label.ndim == 2: label = np.argmax(label, axis=1)
            else:
                label = np.argmax(softmax(feature, axis=1), axis=1)
                # label = np.array([i]*len(feature), np.int32)
            colors.append(label)
        features = np.concatenate(features, axis=0)
        colors = np.concatenate(colors, axis=0)
    
    n_samples, n_features = features.shape
    scaled_features = StandardScaler().fit_transform(features)
    if min(n_samples, n_features) > 64:
        pca_proj = PCA(n_components=64).fit_transform(scaled_features)
    else:
        pca_proj = scaled_features

    if mode == "tsne":
        vis_proj = TSNE(n_jobs=-1).fit_transform(pca_proj)
    elif mode == "umap":
        vis_proj = umap.UMAP(n_jobs=-1).fit_transform(pca_proj)
    else:
        raise NotImplementedError

    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    class_list = np.unique(colors).tolist()
    if n_class is None: n_class = int(max(class_list)) + 1
    palette = np.array(sns.color_palette("hls", n_class))
    plt.figure(figsize=(8,8))
    ax = plt.subplot(aspect='equal')
    scatter_list = []
    for i in class_list:
        c = colors[colors == i]
        t = vis_proj[colors == i, :]
        sc = ax.scatter(t[:, 0], t[:, 1], lw=0, s=40, c=palette[c.astype(np.int32)])
        scatter_list.append(sc)
    plt.legend(scatter_list, class_list, loc="upper right", bbox_to_anchor=(1.1, 1.1), title="Classes")
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight') 
    plt.savefig(f'a_05feature_aggregation/{mode}_visualization.jpg')
    print("Visualization done!")

def radiomic_feature_visualization(img_paths, save_feature_dir, class_name="tumour", mode="tsne", graph=True, n_class=None, features=None, colors=None):
    if features is None or colors is None:
        features, colors = [], []
        for i, img_path in enumerate(img_paths):
            img_name = pathlib.Path(img_path).name.replace(".nii.gz", "")
            logging.info(f"loading feature of {img_name}")
            if graph:
                feature_path = pathlib.Path(f"{save_feature_dir}/{img_name}_{class_name}.json")
                graph_dict = load_json(feature_path)
                feature = np.array(graph_dict["x"])
            else:
                feature_path = pathlib.Path(f"{save_feature_dir}/{img_name}_{class_name}_radiomics.npy")
                feature = np.load(feature_path)
            features.append(feature)
            label = np.argmax(softmax(feature, axis=1), axis=1)
            colors.append(label)
        features = np.concatenate(features, axis=0)
        colors = np.concatenate(colors, axis=0)
    
    n_samples, n_features = features.shape
    scaled_features = StandardScaler().fit_transform(features)
    if min(n_samples, n_features) > 64:
        pca_proj = PCA(n_components=64).fit_transform(scaled_features)
    else:
        pca_proj = scaled_features

    if mode == "tsne":
        vis_proj = TSNE(n_jobs=-1).fit_transform(pca_proj)
    elif mode == "umap":
        vis_proj = umap.UMAP(n_jobs=-1).fit_transform(pca_proj)
    else:
        raise NotImplementedError

    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    class_list = np.unique(colors).tolist()
    if n_class is None: n_class = int(max(class_list)) + 1
    palette = np.array(sns.color_palette("hls", n_class))
    plt.figure(figsize=(8,8))
    ax = plt.subplot(aspect='equal')
    scatter_list = []
    for i in class_list:
        c = colors[colors == i]
        t = vis_proj[colors == i, :]
        sc = ax.scatter(t[:, 0], t[:, 1], lw=0, s=40, c=palette[c.astype(np.int32)])
        scatter_list.append(sc)
    plt.legend(scatter_list, class_list, loc="upper right", bbox_to_anchor=(1.1, 1.1), title="Classes")
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight') 
    plt.savefig(f'a_05feature_aggregation/{mode}_visualization.jpg')
    print("Visualization done!")


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
    parser.add_argument('--resolution', default=20, type=float)
    parser.add_argument('--units', default="power", type=str)
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
            output_list = extract_cnn_pathomic_features(
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






