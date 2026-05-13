#!/bin/bash
cd ~/AE_OfflineRL

echo "=== Baseline CQL 무한 실행 시작 ==="
while true; do
    for seed in 5 6 7 8 9; do
        echo "Baseline CQL Seed $seed 시작..."
        /home/user7/.conda/envs/ad4rl/bin/python main_DDPGCQL.py MA_5LC \
            --dataset highway-humanlike \
            --seed $seed \
            --num-evaluations 10 \
            --project ae_offlinerl \
            --name CQL_baseline_seed${seed}
        echo "Baseline CQL Seed $seed 완료"
    done
    echo "=== 한 사이클 완료, 다시 시작 ==="
done
