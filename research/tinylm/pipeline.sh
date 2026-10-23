#!/bin/bash
# After pretraining (data/tinylm/base) finishes: fine-tune on the personal text for a few
# lengths and replay the typing log with each. Results: data/tinylm/results.txt
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; D=data/tinylm; R=$D/results.txt
while ! grep -q "^.* done$" $D/base/train.log 2>/dev/null; do sleep 60; done
echo "== base" >> $R; $PY eval/replay_log.py --tiny $D/base 2>&1 | grep -A7 "tuned on" >> $R
for n in 500 1000 2000; do
  $PY tinylm/train.py --out $D/ft$n --init $D/base --steps $n --warmup 100 --lr 3e-4 \
      --mix $D/personal_train.bin --mix-frac 0.3 --eval-every 250 > $D/ft$n.out 2>&1
  echo "== ft$n" >> $R; $PY eval/replay_log.py --tiny $D/ft$n 2>&1 | grep -A7 "tuned on" >> $R
done
echo "== all done $(date)" >> $R
