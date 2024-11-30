import sys
sys.path.append('../')

import pathlib
import logging
import argparse

from models.a_04feature_extraction.m_feature_extraction import extract_radiomic_feature
from models.a_05feature_aggregation.m_graph_construction import construct_img_graph

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/TCGA/WSI")
    parser.add_argument('--lab_dir', default="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/TCGA/WSI")
    parser.add_argument('--dataset', default="TCGA-RCC", type=str)
    parser.add_argument('--modality', default="CT", type=str)
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/radiomics", type=str)
    parser.add_argument('--feature_mode', default="SegVol", choices=["pyradiomics", "SegVol"], type=str)
    parser.add_argument('--feature_dim', default=1024, choices=[2048, 384, 1024, 35], type=int)
    parser.add_argument('--resolution', default=1.024, type=float)
    parser.add_argument('--units', default="mm", type=str)
    args = parser.parse_args()

    ## get image and label paths
    class_name = ["kidney_and_mass", "mass", "tumour"][1]
    lab_dir = pathlib.Path(f"{args.lab_dir}/{args.dataset}/{args.modality}")
    lab_paths = lab_dir.rglob(f"{class_name}.nii.gz")
    lab_paths = [f"{p}" for p in lab_paths]
    img_paths = [p.replace(args.lab_dir, args.img_dir) for p in lab_paths]
    img_paths = [p.replace(f"_ensemble/{class_name}.nii.gz", ".nii.gz") for p in img_paths]
    logging.info("The number of images on {}: {}".format(args.dataset, len(img_paths)))
    
    ## set save dir
    save_feature_dir = pathlib.Path(f"{args.save_dir}/{args.dataset}_{args.modality}_radiomic_features/{args.feature_mode}")
    
    # extract radiomics
    # bs = 2
    # nb = len(img_paths) // bs if len(img_paths) % bs == 0 else len(img_paths) // bs + 1
    # for i in range(0, nb):
    #     logging.info(f"Processing images of batch [{i+1}/{nb}] ...")
    #     start = i * bs
    #     end = min(len(img_paths), (i + 1) * bs)
    #     batch_img_paths = img_paths[start:end]
    #     batch_lab_paths = lab_paths[start:end]
    #     extract_radiomic_feature(
    #         img_paths=batch_img_paths,
    #         lab_paths=batch_lab_paths,
    #         feature_mode=args.feature_mode,
    #         save_dir=save_feature_dir,
    #         class_name=class_name,
    #         label=1,
    #         n_jobs=32,
    #         resolution=args.resolution
    #     )

    # construct image graph
    bs = 32
    nb = len(img_paths) // bs if len(img_paths) % bs == 0 else len(img_paths) // bs + 1
    for i in range(0, nb):
        logging.info(f"Processing WSIs of batch [{i+1}/{nb}] ...")
        start = i * bs
        end = min(len(img_paths), (i + 1) * bs)
        batch_img_paths = img_paths[start:end]
        construct_img_graph(
            img_paths=batch_img_paths,
            save_dir=save_feature_dir,
            class_name=class_name,
            n_jobs=8
        )
