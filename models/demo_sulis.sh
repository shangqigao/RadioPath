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
conda activate totalseg

## segment TCIA
python totalseg_tcia.py