#!/bin/bash
#SBATCH --job-name=CLUE
#SBATCH --output=res_panel_comparison.txt
#SBATCH --ntasks=1
#SBATCH --partition=batch

python3 /home-mscluster/tlove/CLUE_SSDP/panel_comparison.py
