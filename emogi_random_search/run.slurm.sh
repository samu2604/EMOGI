sbatch --wait << EOF
#!/bin/bash

#SBATCH -o "$1/slurm_%j.job"
#SBATCH -e "$1/slurm_%j.job"
#SBATCH -J tune
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH -c 4
#SBATCH --mem=15G
#SBATCH -t 06:00:00
#SBATCH --nice=10000


source $HOME/.bashrc
conda activate <your-env>

echo "Called with base directory $1 and budget $2"

<script> -o $1 --config ${1}/config ...
EOF
