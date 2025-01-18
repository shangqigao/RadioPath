#!/bin/bash

#SBATCH -A CRISPIN-ORTUZAR-SL3-GPU
#SBATCH -J radiopath
#SBATCH -o log.%x.job_%j
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-12:00:00
##SBATCH -p cclake
##SBATCH -p cclake-himem
#SBATCH -p ampere
#SBATCH --gres=gpu:1

## activate environment
source ~/.bashrc
conda activate radiopath

## extract pathomics
# wsi_dir="/home/shared/su123/TCGA_ORI"
# save_dir="/home/s/sg2162/projects/Experiments/pathomics"
# python a_06generative_SR/m_pathomics_extraction.py --wsi_dir $wsi_dir --save_dir $save_dir

## pathomic diffusion prior
# wsi_dir="/home/shared/su123/TCGA_ORI"
# save_pathomics_dir="/home/s/sg2162/projects/Experiments/pathomics"
# python a_06generative_SR/m_pathomics_representation.py --wsi_dir $wsi_dir --save_pathomics_dir $save_pathomics_dir

## extract radiomics
# img_dir="/home/s/sg2162/projects/TCIA_NIFTI/image"
# lab_dir="/home/s/sg2162/projects/TCIA_NIFTI/binary_label"
# save_dir="/home/s/sg2162/projects/Experiments/radiomics"
# python a_06generative_SR/m_radiomics_extraction.py --img_dir $img_dir --lab_dir $lab_dir --save_dir $save_dir

## survival analysis
wsi_dir="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/TCGA/WSI"
save_pathomics_dir="/home/sg2162/rds/hpc-work/Experiments/pathomics"
save_radiomics_dir="/home/sg2162/rds/hpc-work/Experiments/radiomics"
save_clinical_dir="/home/sg2162/rds/hpc-work/Experiments/clinical"
python a_07explainable_AI/m_survival_analysis.py --wsi_dir $wsi_dir --save_pathomics_dir $save_pathomics_dir --save_radiomics_dir $save_radiomics_dir --save_clinical_dir $save_clinical_dir

## cancer subtyping
# wsi_dir="/home/shared/su123/TCGA_ORI"
# save_pathomics_dir="/home/s/sg2162/projects/Experiments/pathomics"
# save_clinical_dir="/home/s/sg2162/projects/Experiments/clinical"
# python a_07explainable_AI/m_cancer_subtyping.py --wsi_dir $wsi_dir --save_pathomics_dir $save_pathomics_dir --save_clinical_dir $save_clinical_dir

## concept learning
# wsi_dir="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/TCGA/WSI"
# save_pathomics_dir="/home/sg2162/rds/hpc-work/Experiments/pathomics"
# save_clinical_dir="/home/sg2162/rds/hpc-work/Experiments/clinical"
# python a_07explainable_AI/m_concept_learning.py --wsi_dir $wsi_dir --save_pathomics_dir $save_pathomics_dir --save_clinical_dir $save_clinical_dir