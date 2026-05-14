#!/bin/bash
cd ~/AE_OfflineRL
echo "=== DAE+CQL GPU3 실험 시작 ==="
while true; do
    for dataset in highway-NGSIM highway-final highway-medium highway-random; do
        for seed in 5 6 7 8 9; do
            echo "DAE+CQL dataset=$dataset seed=$seed 시작..."
            CUDA_VISIBLE_DEVICES=3 /home/user7/.conda/envs/ad4rl/bin/python main_DDPGCQL_DAE_v2.py MA_5LC \
                --dataset $dataset \
                --seed $seed \
                --num-evaluations 10 \
                --project ae_offlinerl \
                --group DAE-CQL-highway \
                --name DAE_CQL_${dataset}_seed${seed}
            echo "DAE+CQL dataset=$dataset seed=$seed 완료"
        done
    done
    echo "=== 한 사이클 완료, 다시 시작 ==="
done
