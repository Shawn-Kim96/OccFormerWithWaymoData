#!/bin/bash
#SBATCH --job-name=create_waymo_data
#SBATCH --output=create_waymo_data_%j.log
#SBATCH --partition=cpuql
#SBATCH --time=21-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1

conda activate occformer
cd ~/OccFormerWithWaymoData

python tools/create_data.py waymo \
    --root-path ./data/waymo_v1-3-1 \
    --out-dir ./data/waymo_v1-3-1 \
    --version v1.3.1