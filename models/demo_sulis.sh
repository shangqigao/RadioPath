#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --account=su123

## activate environment
source ~/.bashrc
conda activate radiopath

## test stain normalization
# python a_01stain_normalization/m_stain_normalization.py

## test nucleus instance segmentation
# python a_04feature_extraction/m_nucleus_instance_segmentation.py

## test feature extraction
# python a_04feature_extraction/m_feature_extraction.py

## test graph construction
wsi_dir="/home/shared/su123/TCGA_ORI"
save_dir="/home/s/sg2162/projects/Experiments/pathomics"
python a_06generative_SR/m_pathomics_extraction.py --wsi_dir $wsi_dir --save_dir $save_dir

## test bladder segmentation
# python a_06semantic_segmentation/m_bladder_segmentation.py