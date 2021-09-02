#!/bin/bash
#SBATCH --job-name=CLUE
#SBATCH --output=res_expert_param.txt
#SBATCH --ntasks=1
#SBATCH --partition=batch

python3 /home-mscluster/tlove/CLUE_SSDP/expert_param_test.py
