#!/bin/bash
#SBATCH --job-name=CLUE
#SBATCH --output=res_non_uniform.txt
#SBATCH --ntasks=1
#SBATCH --partition=batch

python3 /home-mscluster/tlove/CLUE_SSDP/nonuniform_test.py
