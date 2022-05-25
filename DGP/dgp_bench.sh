#!/bin/bash
#BATCH -p short-serial
#SBATCH --job-name dgp_bench
#SBATCH -o dgp_bench-%j.o
#SBATCH -e dgp_bench-%j.e
#SBATCH --open-mode=truncate
#SBATCH -t 12:00:00
#SBATCH --mem=50GB

python -u dgp_bench.py
