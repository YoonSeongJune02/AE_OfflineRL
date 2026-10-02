#!/bin/bash
# DAE+CQL 의도 검증 전체 실행
#
#   GPUS="<번호> <번호>" bash verify/run_all.sh all      # 1 → 2 → 3 → 4 순서대로 전부
#   bash verify/run_all.sh 1        # 검증 1만 (DAE 학습 + OOD 감지 + 패널티 신호)
#   bash verify/run_all.sh 2        # 정책 학습만 (SUMO 없음)
#   bash verify/run_all.sh 3        # 평가만 (SUMO 사용)
#   bash verify/run_all.sh 4        # 집계만
#
# 오래 걸리니 보통은 이렇게 백그라운드로:
#   GPUS="<번호> <번호>" nohup bash verify/run_all.sh all > verify_out_run.log 2>&1 &
#
# GPUS는 서버 상황(nvidia-smi)을 보고 그때마다 정한다. 지정하지 않으면 실행하지 않는다.
#
# 아래 값만 바꿔서 쓰면 된다.

set -u
cd "$(dirname "$0")/.."

PY=${PY:-/home/user7/.conda/envs/ad4rl/bin/python}
DATASETS=(${DATASETS:-highway-NGSIM highway-final-medium highway-humanlike})
SEEDS=(${SEEDS:-5 6 7})
PRESETS=(${PRESETS:-cql dae_orig dae_fix})
GPUS=(${GPUS:-})
CONDITIONS=${CONDITIONS:-"clean noise noise_orig denoise cutin lanereduction"}
EPISODES=${EPISODES:-5}
MAX_PAR_TRAIN=${MAX_PAR_TRAIN:-4}   # 학습은 SUMO를 안 쓰므로 여러 개 동시에 돌려도 된다
MAX_PAR_EVAL=${MAX_PAR_EVAL:-2}     # 평가는 SUMO를 띄우므로 적게
V1_ARGS=${V1_ARGS:-}                # 예: V1_ARGS="--dae-epochs 3" (빠른 시험용)
TRAIN_ARGS=${TRAIN_ARGS:-}          # 예: TRAIN_ARGS="--epochs 1 --itr 2000" (빠른 시험용)
EVAL_ARGS=${EVAL_ARGS:-}

if [ "${1:-all}" != "4" ] && [ ${#GPUS[@]} -eq 0 ]; then
  echo "GPUS를 지정하세요. 예: GPUS=\"<번호> <번호>\" bash verify/run_all.sh all"
  exit 1
fi

LOG_DIR=verify_out/logs
mkdir -p "$LOG_DIR"

gpu_i=0
next_gpu() { echo "${GPUS[$((gpu_i % ${#GPUS[@]}))]}"; gpu_i=$((gpu_i + 1)); }

# 동시에 도는 작업 수를 제한한다
throttle() {
  local max=$1
  while [ "$(jobs -rp | wc -l)" -ge "$max" ]; do sleep 5; done
}

step1() {
  echo "=== [1] DAE 학습 + OOD 감지 + 패널티 신호 ($(date '+%m-%d %H:%M')) ==="
  for ds in "${DATASETS[@]}"; do
    for s in "${SEEDS[@]}"; do
      throttle "$MAX_PAR_TRAIN"
      g=$(next_gpu)
      echo "  v1 $ds seed$s (GPU $g)"
      CUDA_VISIBLE_DEVICES=$g $PY verify/v1_dae_ood.py --dataset "$ds" --seed "$s" $V1_ARGS \
        > "$LOG_DIR/v1_${ds}_seed${s}.log" 2>&1 &
    done
  done
  wait
  echo "  → verify_out/results/v1_parts/, v1_hist_*.png"
}

step2() {
  echo "=== [2] 정책 학습 ($(date '+%m-%d %H:%M')) ==="
  for ds in "${DATASETS[@]}"; do
    for s in "${SEEDS[@]}"; do
      for p in "${PRESETS[@]}"; do
        if [ -f "verify_out/checkpoints/policy/${p}_${ds}_seed${s}/actor_final.pt" ]; then
          echo "  건너뜀 (이미 있음): ${p}_${ds}_seed${s}"
          continue
        fi
        throttle "$MAX_PAR_TRAIN"
        g=$(next_gpu)
        echo "  train $p $ds seed$s (GPU $g)"
        CUDA_VISIBLE_DEVICES=$g $PY verify/train_policy.py --preset "$p" --dataset "$ds" --seed "$s" $TRAIN_ARGS \
          > "$LOG_DIR/train_${p}_${ds}_seed${s}.log" 2>&1 &
      done
    done
  done
  wait
  echo "  → verify_out/checkpoints/policy/"
}

step3() {
  echo "=== [3] 평가 ($(date '+%m-%d %H:%M')) ==="
  for ds in "${DATASETS[@]}"; do
    for s in "${SEEDS[@]}"; do
      throttle "$MAX_PAR_EVAL"
      g=$(next_gpu)
      echo "  eval $ds seed$s (GPU $g)"
      # 평가마다 다른 파일에 쓰고 나중에 합친다 (동시에 한 파일에 쓰지 않도록)
      rm -f "verify_out/results/eval_parts/${ds}_seed${s}.csv"
      CUDA_VISIBLE_DEVICES=$g $PY verify/evaluate.py --dataset "$ds" --seed "$s" \
        --presets ${PRESETS[*]} --conditions $CONDITIONS --episodes "$EPISODES" $EVAL_ARGS \
        --out "verify_out/results/eval_parts/${ds}_seed${s}.csv" \
        > "$LOG_DIR/eval_${ds}_seed${s}.log" 2>&1 &
      sleep 20   # SUMO가 동시에 뜨면서 포트가 겹치지 않도록 간격을 둔다
    done
  done
  wait
  merge_eval
}

merge_eval() {
  local out=verify_out/results/eval_episodes.csv
  local first=1
  rm -f "$out"
  for f in verify_out/results/eval_parts/*.csv; do
    [ -f "$f" ] || continue
    if [ $first -eq 1 ]; then cat "$f" > "$out"; first=0; else tail -n +2 "$f" >> "$out"; fi
  done
  echo "  → $out ($(($(wc -l < "$out") - 1))개 에피소드)"
}

step4() {
  echo "=== [4] 집계 ($(date '+%m-%d %H:%M')) ==="
  [ -d verify_out/results/eval_parts ] && merge_eval
  $PY verify/summarize.py
}

case "${1:-all}" in
  1) step1 ;;
  2) step2 ;;
  3) step3 ;;
  4) step4 ;;
  all) step1; step2; step3; step4 ;;
  *) echo "사용법: bash verify/run_all.sh [1|2|3|4|all]"; exit 1 ;;
esac
echo "=== 끝 ($(date '+%m-%d %H:%M')) ==="
