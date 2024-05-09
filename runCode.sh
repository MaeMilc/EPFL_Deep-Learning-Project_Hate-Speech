#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 4G
#SBATCH --time 24:00:00
#SBATCH --gres gpu:1
#SBATCH --qos ee-559

python3 bertweet_base.py

