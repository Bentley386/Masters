#!/bin/bash
#SBATCH --time=0-5:00:00
#SBATCH --mem=4096

ml Anaconda3
date +"%T"
python prehodiPoStanjih.py $1
date +"%T" 
