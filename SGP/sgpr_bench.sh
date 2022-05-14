#!/bin/bash
#BATCH -p short-serial
#SBATCH --job-name sgpr_bench
#SBATCH -o sgpr_bench-%j.o
#SBATCH -e sgpr_bench-%j.e
#SBATCH --open-mode=truncate
#SBATCH -t 12:00:00
#SBATCH --mem=40GB

python -u sgpr_bench.py
