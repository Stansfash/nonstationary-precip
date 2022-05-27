#!/bin/bash
#SBATCH --account=gpu
#SBATCH --partition=gpu
#SBATCH --nodelist=node022
#SBATCH --job-name dgp_bench
#SBATCH -o dgp_bench-%j.o
#SBATCH -e dgp_bench-%j.e
#SBATCH --open-mode=truncate
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20GB

python -u dgp_bench.py
