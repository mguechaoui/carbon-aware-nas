#!/usr/bin/env bash
# Sequential overnight run. Keep the laptop plugged in, set Windows power plan to "never sleep", close other apps.
# Usage:  nohup bash run_overnight.sh > overnight.log 2>&1 &     then: tail -f overnight.log
set -u
ts=$(date +%Y%m%d_%H%M)
mkdir -p backup_$ts && cp -r results models backup_$ts/ 2>/dev/null
rm -rf results/nas.db results/*.json results/*.png models/model_*
mkdir -p results logs

run() {
  echo "=== $(date) START: $*"
  "$@" > "logs/$(echo $1_$2 | tr -c 'a-zA-Z0-9_\n' _).log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then echo "=== $(date) END ok: $*"; else echo "=== $(date) FAILED (exit $rc): $*  -> see logs/"; fi
}

run python energy_model.py --n 360            # 1) hardware characterization of the whole space + predictor
run python nas.py --trials 60 --epochs 3      # 2) multi-objective search
run python pareto.py --k 6                    # 3) Pareto front + selection
run python finalize.py --epochs 20            # 4) final models, test accuracy, energy, switch cost
run python switcher.py                        # 5) runtime simulation
run python proxy_study.py --n 24 --short 2 --long 8   # 6) optional: proxy reliability (longest; resumable)
echo "=== ALL DONE $(date)"