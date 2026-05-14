#!/bin/bash
cd ~/AE_OfflineRL
echo "=== Baseline CQL GPU2 실험 시작 ==="
while true; do
    for dataset in highway-NGSIM highway-final highway-medium highway-random; do
        for seed in 5 6 7 8 9; do
            echo "Baseline CQL dataset=$dataset seed=$seed 시작..."
            /home/user7/.conda/envs/ad4rl/bin/python main_DDPGCQL.py MA_5LC \
                --dataset $dataset \
                --seed $seed \
                --num-evaluations 10 \
                --project ae_offlinerl \
                --group CQL-baseline-highway \
                --name CQL_${dataset}_seed${seed}
            echo "Baseline CQL dataset=$dataset seed=$seed 완료"
        done
    done
    echo "=== 한 사이클 완료, 다시 시작 ==="
done
