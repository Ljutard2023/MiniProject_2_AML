#!/bin/bash
#BSUB -q gpuv100
#BSUB -J AML_geodesics
#BSUB -n 4
#BSUB -gpu "num=1"
#BSUB -W 1:00
#BSUB -R "rusage[mem=8GB] span[hosts=1]"
#BSUB -o logs/geodesics_%J.log
#BSUB -e logs/geodesics_%J.err

source /zhome/3f/9/223204/projects/ADLCV/adlcv-ex2_1/.venv/bin/activate

python ensemble_vae.py geodesics \
    --device cuda \
    --experiment-folder exp \
    --num-curves 25 \
    --num-t 20 \
    --geodesic-steps 500 \
    --geodesic-lr 1e-3
