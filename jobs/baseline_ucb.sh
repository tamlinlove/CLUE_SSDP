#!/bin/bash
#SBATCH --job-name=CLUE
#SBATCH --output=res_beta_param.txt
#SBATCH --ntasks=1
#SBATCH --partition=batch

python3 /home-mscluster/tlove/CLUE_SSDP/baseline_test.py "UCB"
