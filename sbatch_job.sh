#!/bin/bash

#SBATCH --job-name=pathformer               # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:1                          # Using 1 gpu
#SBATCH --time=0-12:00:00                     # 1 hour timelimit
#SBATCH --mem=20000MB                         # Using 10GB CPU Memory
#SBATCH --cpus-per-task=2                     # Using 4 maximum processor

source /home/s3/${USER}/.bashrc
source /home/s3/${USER}/anaconda3/bin/activate
conda activate lisa_env
# srun sh scripts/multivariate/electricity.sh
# srun sh scripts/multivariate/ETTh1.sh
# srun sh scripts/multivariate/ETTh2.sh
# srun sh scripts/multivariate/ETTm1.sh
# srun sh scripts/multivariate/ETTm2.sh
# srun sh scripts/multivariate/ill.sh
# srun sh scripts/multivariate/traffic.sh
srun sh scripts/multivariate/weather.sh
