#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64g
#SBATCH -p academic
#SBATCH -t 30:00:00
#SBATCH --gres=gpu:1

RUN_NAME=$1

module load python/3.10.17
module load cuda/12.4.0/3mdaov5

python3 -m venv pytorch_venv
source pytorch_venv/bin/activate
pip3 install -r requirements.txt
pip3 install -e Modules/4D-Humans/.
pip3 install -e Grounded-SAM-2/.

python3 Code/generate_scenes.py --run_name $RUN_NAME
