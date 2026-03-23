#!/bin/bash
#BSUB -q gpuv100
#BSUB -J AML_cov
#BSUB -n 4
#BSUB -gpu "num=1"
#BSUB -W 8:00
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -o logs/cov_%J.log
#BSUB -e logs/cov_%J.err

source /zhome/3f/9/223204/projects/ADLCV/adlcv-ex2_1/.venv/bin/activate

python ensemble_vae.py compute_cov \
    --device cuda \
    --experiment-folder exp \
    --num-reruns 10 \
    --num-pairs-cov 10 \
    --num-t 20 \
    --geodesic-steps 300 \
    --geodesic-lr 1e-2 \
    --epochs-per-decoder 50 \
    --batch-size 32 \
    --latent-dim 2
