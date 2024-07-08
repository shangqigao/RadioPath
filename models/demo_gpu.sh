#!/bin/bash

#SBATCH -A CRISPIN-ORTUZAR-SL3-GPU
#SBATCH -J radiopath
#SBATCH -o log.%x.job_%j
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:00:00
##SBATCH -p cclake
##SBATCH -p cclake-himem
#SBATCH -p ampere
#SBATCH --gres=gpu:1

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
python a_06generative_SR/m_pathomics_extraction.py

## test bladder segmentation
# python a_06semantic_segmentation/m_bladder_segmentation.py