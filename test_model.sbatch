#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --account=dsci410_510
#SBATCH --partition=gpu
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err
#SBATCH --time=1:00:00

module purge
module load miniconda3/20240410
conda activate SpectrogramClassifierEnv


python run_model.py --chkpt /projects/dsci410_510/Kolahi_models/checkpoints/epoch=11-val_loss=0.000-val_acc=1.000.ckpt

