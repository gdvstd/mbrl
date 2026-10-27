#!/bin/bash
# Gate-design ablation on top of the online-KS tau (the adopted ID term),
# 3 seeds x both scenarios x 4 variants, chained after saekfull:
#  ksent     KS ∧ entropy(eta=.5) ∧ clock-coin   (oracle-screen winner)
#  ksdis     KS ∧ disagree(kappa=.5) ∧ clock-coin (screen predicts HARMFUL
#            in altgoal -- included to validate the screen's predictions)
#  kssucc    KS ∧ success-annealed coin          (competence-based decay)
#  ksentsucc KS ∧ entropy ∧ success-annealed coin (combined)
# Reference baselines already run: saeks (KS ∧ clock-coin), ebtl.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
mkdir -p runs/logs
until grep -q V4_SAEKFULL_DONE runs/logs/v4saekfull_all.log 2>/dev/null; do
  sleep 300
done
for tr in ksent ksdis kssucc ksentsucc; do
  for s in 0 1 2; do
    for sc in altgoal locked; do
      $PY fourroom/train_paper_student.py --scenario $sc --transfer $tr \
          --seed $s --teacher runs/paperv4_${sc}_teacher/final_model.zip \
          --outdir runs/paperv4_${sc}_${tr}_seed$s \
          > runs/logs/v4_${tr}_${sc}_s$s.log 2>&1 &
    done
  done
  wait
  echo "V4_${tr}_DONE"
done
echo V4_KSGATES_DONE
