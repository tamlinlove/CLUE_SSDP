#!/bin/bash
#SBATCH --job-name=CLUE
#SBATCH --output=res_adversarial.txt
#SBATCH --ntasks=1
#SBATCH --partition=batch

python3 /home-mscluster/tlove/CLUE_SSDP/adversarial_test.py
