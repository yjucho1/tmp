#!/bin/bash

#SBATCH --job-name=pathformer               # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:1                          # Using 1 gpu
#SBATCH --time=0-12:00:00                     # 1 hour timelimit
#SBATCH --mem=50000MB                         # Using 10GB CPU Memory
#SBATCH --cpus-per-task=2                     # Using 4 maximum processor

source /home/s3/${USER}/.bashrc
source /home/s3/${USER}/anaconda3/bin/activate
conda activate lisa_env
# srun sh scripts/autoregressive/electricity.sh
# srun sh scripts/autoregressive/ETTh1.sh
# srun sh scripts/autoregressive/ETTh2.sh
# srun sh scripts/autoregressive/ETTm1.sh
srun sh scripts/autoregressive/ETTm2.sh
# srun sh scripts/autoregressive/ill.sh
# srun sh scripts/autoregressive/traffic.sh
# srun sh scripts/autoregressive/weather.sh
