#!/bin/bash
# wait for all stage-B runs, evaluate the missing ones, build tables/figures
export OMP_NUM_THREADS=1
until [ $(ls ../results/v2B/*.pt 2>/dev/null | wc -l) -ge 12 ]; do
  for r in ../results/v2B/v2_*.pt; do b=$(basename $r .pt); [ -f ../results/v2B/evalA_$b.json ] || [ -f ../results/v2B/.busy_$b ] || { touch ../results/v2B/.busy_$b; (python3 eval_A.py $r ../results/v2B/evalA_$b.json > ../results/v2B/evalA_$b.log 2>&1; rm -f ../results/v2B/.busy_$b) & }; done
  sleep 60
done
for r in ../results/v2B/v2_*.pt; do b=$(basename $r .pt); [ -f ../results/v2B/evalA_$b.json ] || [ -f ../results/v2B/.busy_$b ] || { touch ../results/v2B/.busy_$b; (python3 eval_A.py $r ../results/v2B/evalA_$b.json > ../results/v2B/evalA_$b.log 2>&1; rm -f ../results/v2B/.busy_$b) & }; done
wait; sleep 5
while ls ../results/v2B/.busy_* >/dev/null 2>&1; do sleep 10; done
python3 make_report_v2.py > ../results/finish_v2.log 2>&1; echo finished >> ../results/finish_v2.log
