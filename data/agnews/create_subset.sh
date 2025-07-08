#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --partition=cpu2022
#SBATCH --time=0:30:00
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log

export PYTHONUNBUFFERED=1

module load conda
conda activate CLVQVAE_same

python create_subset.py