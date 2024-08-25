import sys
sys.path.append('../')

import pathlib
import logging
import torch
import argparse

from common.m_utils import select_wsi

from models.a_02tissue_masking.m_tissue_masking import generate_wsi_tissue_mask
from models.a_04feature_extraction.m_feature_extraction import extract_pathomic_feature
from models.a_05feature_aggregation.m_graph_construction import construct_wsi_graph
from models.a_05feature_aggregation.m_graph_construction import visualize_graph
from models.a_05feature_aggregation.m_graph_construction import feature_visualization
from models.a_05feature_aggregation.m_graph_construction import generate_node_label
from models.a_05feature_aggregation.m_graph_construction import extract_minimum_spanning_tree
from models.a_06generative_SR.m_zeroshot_classification import pathology_zeroshot_classification, load_prompts

torch.multiprocessing.set_sharing_strategy("file_system")

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', default="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/TCGA/WSI")
    parser.add_argument('--dataset', default="TCGA-RCC", type=str)
    parser.add_argument('--prompts', default="./a_06generative_SR/TCGA_prompts.json", type=str)
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/pathomics", type=str)
    parser.add_argument('--mask_method', default='otsu', choices=["otsu", "morphological"], help='method of tissue masking')
    parser.add_argument('--mode', default="wsi", choices=["tile", "wsi"], type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--feature_mode', default="vit", choices=["cnn", "vit", "uni", "conch"], type=str)
    parser.add_argument('--node_features', default=1024, choices=[2048, 384, 1024, 35], type=int)
    parser.add_argument('--resolution', default=20, type=float)
    parser.add_argument('--units', default="power", type=str)
    args = parser.parse_args()

    ## get wsi path
    wsi_dir = pathlib.Path(args.wsi_dir) / args.dataset
    excluded_wsi = ["TCGA-5P-A9KC-01Z-00-DX1", "TCGA-5P-A9KA-01Z-00-DX1"]
    wsi_paths = select_wsi(wsi_dir, excluded_wsi)
    logging.info("The number of selected WSIs on {}: {}".format(args.dataset, len(wsi_paths)))
    
    ## set save dir
    save_tile_dir = pathlib.Path(f"{args.save_dir}/{args.dataset}_pathomic_tiles")
    save_msk_dir = pathlib.Path(f"{args.save_dir}/{args.dataset}_{args.mode}_pathomic_masks")
    save_feature_dir = pathlib.Path(f"{args.save_dir}/{args.dataset}_{args.mode}_pathomic_features/{args.feature_mode}")
    save_classification_dir = pathlib.Path(f"{args.save_dir}/{args.dataset}_{args.mode}_pathomic_features/conch")
    save_model_dir = pathlib.Path(f"{args.save_dir}/{args.dataset}_{args.mode}_models/{args.feature_mode}")
    

    # generate wsi tissue mask batch by batch
    # if args.mode == "wsi":
    #     bs = 32
    #     nb = len(wsi_paths) // bs if len(wsi_paths) % bs == 0 else len(wsi_paths) // bs + 1
    #     for i in range(0, nb):
    #         logging.info(f"Processing WSIs of batch [{i+1}/{nb}] ...")
    #         start = i * bs
    #         end = min(len(wsi_paths), (i + 1) * bs)
    #         batch_wsi_paths = wsi_paths[start:end]
    #         generate_wsi_tissue_mask(
    #             wsi_paths=batch_wsi_paths,
    #             save_msk_dir=save_msk_dir,
    #             n_jobs=32,
    #             method=args.mask_method,
    #             resolution=1.25,
    #             units="power"
    #         )

    # extract wsi feature patch by patch
    # if args.mode == "wsi":
    #     msk_paths = [save_msk_dir / f"{p.stem}.jpg" for p in wsi_paths]
    #     logging.info("The number of extracted tissue masks on {}: {}".format(args.dataset, len(msk_paths)))
    # else:
    #     msk_paths = None
    # if args.mode == "wsi":
    #     bs = 32
    #     nb = len(wsi_paths) // bs if len(wsi_paths) % bs == 0 else len(wsi_paths) // bs + 1
    #     for i in range(20, nb):
    #         logging.info(f"Processing WSIs of batch [{i+1}/{nb}] ...")
    #         start = i * bs
    #         end = min(len(wsi_paths), (i + 1) * bs)
    #         batch_wsi_paths = wsi_paths[start:end]
    #         batch_msk_paths = msk_paths[start:end]
    #         extract_pathomic_feature(
    #             wsi_paths=batch_wsi_paths,
    #             wsi_msk_paths=batch_msk_paths,
    #             feature_mode=args.feature_mode,
    #             save_dir=save_feature_dir,
    #             mode=args.mode,
    #             resolution=args.resolution,
    #             units=args.units
    #         )

    # zero-shot classification
    # if args.mode == "wsi":
    #     msk_paths = [save_msk_dir / f"{p.stem}.jpg" for p in wsi_paths]
    #     logging.info("The number of extracted tissue masks on {}: {}".format(args.dataset, len(msk_paths)))
    # else:
    #     msk_paths = None
    # prompts = load_prompts(args.prompts)
    # if args.mode == "wsi":
    #     bs = 32
    #     nb = len(wsi_paths) // bs if len(wsi_paths) % bs == 0 else len(wsi_paths) // bs + 1
    #     for i in range(29, nb):
    #         logging.info(f"Processing WSIs of batch [{i+1}/{nb}] ...")
    #         start = i * bs
    #         end = min(len(wsi_paths), (i + 1) * bs)
    #         batch_wsi_paths = wsi_paths[start:end]
    #         batch_msk_paths = msk_paths[start:end]
    #         pathology_zeroshot_classification(
    #             wsi_paths=batch_wsi_paths,
    #             wsi_msk_paths=batch_msk_paths,
    #             cls_mode=args.feature_mode,
    #             save_dir=save_feature_dir,
    #             mode=args.mode,
    #             prompts=prompts,
    #             resolution=args.resolution,
    #             units=args.units
    #         )

    # construct wsi graph
    # bs = 32
    # nb = len(wsi_paths) // bs if len(wsi_paths) % bs == 0 else len(wsi_paths) // bs + 1
    # for i in range(0, nb):
    #     logging.info(f"Processing WSIs of batch [{i+1}/{nb}] ...")
    #     start = i * bs
    #     end = min(len(wsi_paths), (i + 1) * bs)
    #     batch_wsi_paths = wsi_paths[start:end]
    #     construct_wsi_graph(
    #         wsi_paths=batch_wsi_paths,
    #         save_dir=save_feature_dir,
    #         n_jobs=8
    #     )

    # extract minimum spanning tree
    # wsi_graph_paths = [save_feature_dir / f"{p.stem}.json" for p in wsi_paths]
    # extract_minimum_spanning_tree(
    #     wsi_graph_paths=wsi_graph_paths,
    #     save_dir=save_feature_dir,
    #     n_jobs=8
    # )

    # label graph node
    # wsi_cls_paths = [save_classification_dir / f"{p.stem}.features.npy" for p in wsi_paths]
    # wsi_graph_paths = [save_feature_dir / f"{p.stem}.json" for p in wsi_paths]
    # generate_node_label(
    #     wsi_paths=wsi_paths,
    #     wsi_annot_paths=wsi_cls_paths,
    #     wsi_graph_paths=wsi_graph_paths,
    #     save_lab_dir=save_feature_dir,
    #     anno_type="classification",
    #     n_jobs=8
    # )

    # visualize feature
    # feature_visualization(
    #     wsi_paths=wsi_paths[0:900:90],
    #     save_feature_dir=save_feature_dir,
    #     mode="umap",
    #     save_label_dir=None,
    #     graph=True,
    #     num_class=args.node_features
    # )


    ## visualize graph on wsi
    wsi_path = wsi_paths[2]
    wsi_name = pathlib.Path(wsi_path).stem 
    logging.info(f"Visualizing graph of {wsi_name}...")
    graph_path = save_feature_dir / f"{wsi_name}.json"
    label_path = save_feature_dir / f"{wsi_name}.label.npy"
    subgraph_id = 32
    if subgraph_id is not None: 
        prompts = load_prompts(args.prompts, index=0)
        class_name = prompts[subgraph_id]
        logging.info(f"Visualizing subgraph of {class_name}...")
    visualize_graph(
        wsi_path=wsi_path,
        graph_path=graph_path,
        label=label_path,
        subgraph_id=subgraph_id,
        show_map=False,
        magnify=True,
        resolution=args.resolution,
        units=args.units
    )