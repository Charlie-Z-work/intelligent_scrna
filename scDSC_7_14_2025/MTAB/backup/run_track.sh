#!/bin/bash
#SBATCH --job-name=track_best
#SBATCH --account=yzhou_351_0001
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=track_result.out

module load python/3.11.11/pytorch/2025.04.01
cd /users/tzheng/scDSC/scDSC_7_13_2025/MTAB

python track_best_config.py
