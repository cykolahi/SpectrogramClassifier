#!/bin/bash
#SBATCH --account=dsci410_510
#SBATCH --job-name=train_SpectrogramClassifier
#SBATCH --output=train_SpectrogramClassifier.out
#SBATCH --error=train_SpectrogramClassifier.err
#SBATCH --partition=compute
#SBATCH --time=0-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G

module load miniconda3/20240410
conda activate dlproject_v4

#python train_model.py
python train_model.py



