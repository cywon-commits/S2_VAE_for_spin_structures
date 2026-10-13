#!/bin/bash
# Main sweep (CPU, 4 parallel single-thread jobs).  usage: run_all.sh ALPHA STEPS   (run from src/)
A=${1:-1e-5}; ST=${2:-2500}; export OMP_NUM_THREADS=1; mkdir -p ../results/runs ../results/logs
{
 # R=1 (global latent), beta=1, all temperatures
 for J in "0.80 o" "0.95 o" "1.05 o" "1.10 o" "1.12 o" "1.12 r" "1.14 r" "1.18 r" "1.30 r"; do echo "$J 1 1"; done
 # spatially resolved latents, beta=1
 for R in 2 3 4; do for J in "0.80 o" "0.95 o" "1.05 o" "1.10 o" "1.12 o"; do echo "$J $R 1"; done; done
 # beta sweep, global latent
 for B in 0.5 2 4; do for J in "0.95 o" "1.10 o"; do echo "$J 1 $B"; done; done
} | xargs -P 4 -L 1 bash -c 'python3 train.py --T $0 --st $1 --R $2 --beta $3 --alpha '$A' --steps '$ST' --bs 4 --out ../results/runs > ../results/logs/T$0$1_R$2_b$3.log 2>&1'
