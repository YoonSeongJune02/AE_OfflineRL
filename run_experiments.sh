#!/bin/bash
# CQL(baseline) / DAE+CQL 학습 실행
#
#   ALGO=cql GPU=2 bash run_experiments.sh
#   ALGO=dae GPU=3 DATASETS="highway-NGSIM" SEEDS="5 6 7" EXTRA="--shape-lambda-max 1.0" bash run_experiments.sh
#
# 백그라운드로 돌릴 때:
#   ALGO=cql GPU=2 nohup bash run_experiments.sh > run_cql.log 2>&1 &
#
# 데이터셋 × seed 조합을 한 번씩 차례로 돌리고 끝난다.

set -u
cd "$(dirname "$0")"

ALGO=${ALGO:-}
GPU=${GPU:-}
PY=${PY:-/home/user7/.conda/envs/ad4rl/bin/python}
SCENARIO=${SCENARIO:-MA_5LC}
DATASETS=(${DATASETS:-highway-NGSIM highway-final highway-medium highway-random highway-final-medium highway-final-random highway-humanlike})
SEEDS=(${SEEDS:-5 6 7 8 9})
NUM_EVAL=${NUM_EVAL:-5}
PROJECT=${PROJECT:-ae_offlinerl}
EXTRA=${EXTRA:-}

case "$ALGO" in
  cql) MAIN=main_DDPGCQL.py;        PREFIX=CQL;     GROUP=${GROUP:-CQL-baseline-highway} ;;
  dae) MAIN=main_DDPGCQL_DAE_v2.py; PREFIX=DAE_CQL; GROUP=${GROUP:-DAE-CQL-highway} ;;
  *) echo "ALGO를 cql 또는 dae로 지정하세요. 예: ALGO=cql GPU=2 bash run_experiments.sh"; exit 1 ;;
esac

if [ -z "$GPU" ]; then
  echo "GPU 번호를 지정하세요. 예: ALGO=$ALGO GPU=2 bash run_experiments.sh"
  exit 1
fi

for ds in "${DATASETS[@]}"; do
  for s in "${SEEDS[@]}"; do
    name=${PREFIX}_${ds}_seed${s}
    echo "[$(date '+%m-%d %H:%M')] 시작: $name (GPU $GPU)"
    CUDA_VISIBLE_DEVICES=$GPU $PY $MAIN $SCENARIO \
      --dataset "$ds" \
      --seed "$s" \
      --num-evaluations "$NUM_EVAL" \
      --project "$PROJECT" \
      --group "$GROUP" \
      --name "$name" \
      $EXTRA
    echo "[$(date '+%m-%d %H:%M')] 끝: $name (exit $?)"
  done
done
