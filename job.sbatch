#!/bin/bash
#SBATCH --account=dsci410_510
#SBATCH --job-name=DLproj
#SBATCH --output=DLproj.out
#SBATCH --error=DLproj.err
#SBATCH --partition=compute
#SBATCH --time=0-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

module load miniconda3/20240410
conda activate dlproject_v4

#python train_model.py
srun python train_model.py



