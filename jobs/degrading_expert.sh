#!/bin/bash
#SBATCH --job-name=CLUE
#SBATCH --output=res_degrading_expert.txt
#SBATCH --ntasks=1
#SBATCH --partition=batch

python3 /home-mscluster/tlove/CLUE_SSDP/degrading_expert_test.py
