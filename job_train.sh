#!/bin/bash
#BSUB -q gpuv100
#BSUB -J AML_train
#BSUB -n 4
#BSUB -gpu "num=1"
#BSUB -W 1:00
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -o logs/train_%J.log
#BSUB -e logs/train_%J.err

source /zhome/3f/9/223204/projects/ADLCV/adlcv-ex2_1/.venv/bin/activate

python ensemble_vae.py train \
    --device cuda \
    --experiment-folder exp \
    --epochs-per-decoder 50 \
    --batch-size 32 \
    --latent-dim 2
