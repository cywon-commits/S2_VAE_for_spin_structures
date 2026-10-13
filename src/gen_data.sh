#!/bin/bash
# usage: gen_data.sh  (run from data/)
export PYTHONPATH=../src OMP_NUM_THREADS=1
printf "0.80 o\n0.95 o\n1.05 o\n1.10 o\n1.12 o\n1.12 r\n1.14 r\n1.18 r\n1.30 r\n" | xargs -P 3 -L 1 bash -c 'python3 ../src/ll_mc.py 12 $0 $1 2000 40000 20 >> gen.log'
