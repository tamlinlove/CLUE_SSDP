#!/bin/bash
#SBATCH --job-name=CLUE
#SBATCH --output=res.txt
#SBATCH --ntasks=1
#SBATCH --partition=batch

python3 /home-mscluster/tlove/CLUE_SSDP/tests.py
