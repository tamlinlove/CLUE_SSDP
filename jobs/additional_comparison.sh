#!/bin/bash
#SBATCH --job-name=CLUE
#SBATCH --output=res_additional.txt
#SBATCH --ntasks=1
#SBATCH --partition=batch

python3 /home-mscluster/tlove/CLUE_SSDP/additional_comparisons_test.py
