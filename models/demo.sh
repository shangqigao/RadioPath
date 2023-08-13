#!/bin/bash

#SBATCH --job-name=KiBla        # job name (default is the name of this file)
#SBATCH --output=log.%x.job_%j  # file name for stdout/stderr (%x will be replaced with the job name, %j with the jobid)
#SBATCH --time=4:00:00          # maximum wall time allocated for the job (D-H:MM:SS)
#SBATCH --partition=gpu_short   # put the job into the gpu partition
#SBATCH --mem=20G               # RAM per node
#SBATCH --threads-per-core=1    # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task=4       # number of CPUs per process

## nodes allocation
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # MPI processes per node

## GPU allocation
#SBATCH --gres=gpu:1            # number of GPUs per node (gres=gpu:N)

## activate environment
source ~/.bashrc
conda activate tiatoolbox-dev

## test stain normalization
# python a_01stain_normalization/m_stain_normalization.py

## test nucleus instance segmentation
# python a_04feature_extraction/m_nucleus_instance_segmentation.py

## test feature extraction
# python a_04feature_extraction/m_feature_extraction.py

## test graph construction
python a_05feature_aggregation/m_graph_construction.py

## test bladder segmentation
# python a_06semantic_segmentation/m_bladder_segmentation.py