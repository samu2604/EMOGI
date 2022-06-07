#!/bin/bash

#SBATCH -o "slurm_out_%j.job"
#SBATCH -e "slurm_err_%j.job"
#SBATCH -J plurimut6h
#SBATCH -p cpu_p
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 12:00:00
#SBATCH --nice=10000


source $HOME/.bashrc
conda activate emogi_env

echo "Called with base directory $1 and budget $2"

python $HOME/EMOGI/EMOGI/train_EMOGI_cv.py --config=config

