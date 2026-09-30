# DAE+CQL 의도 검증

검증하려는 주장은 이것 하나다.

> DAE 패널티로 학습한 정책은 데이터 분포 안쪽(익숙한 상태)에 머무르려 하고,
> 그래서 낯선 상황에서 CQL보다 안전하게 행동한다.

이 주장을 세 단계로 나눠서 확인한다.

| 항목 | 질문 | 스크립트 | 보는 값 | 주장을 지지하려면 |
| --- | --- | --- | --- | --- |
| 검증 1 | DAE가 OOD 상태를 구분하는가 | `v1_dae_ood.py` | 정상 held-out 대비 AUROC | 노이즈·다른 시나리오에서 AUROC ≥ 0.8 |
| 검증 1b | 학습 중 패널티가 state에 따라 일관되게 걸렸는가 | `v1_dae_ood.py` | 같은 state 반복 시 패널티 상관, 패널티 크기 / reward σ | 반복 상관이 높고, 패널티가 reward 차이를 덮지 않음 |
| 검증 2 | 정책이 실제로 익숙한 상태에 머무르는가 | `train_policy.py` → `evaluate.py` | 방문 state의 재구성오차 (같은 측정용 DAE) | DAE+CQL < CQL, 짝지은 차이가 0을 벗어남 |
| 검증 3 | 그 결과 낯선 조건에서 더 안전한가 | `evaluate.py` | correction_reward, 조기 종료 비율 | 노이즈·다른 시나리오에서 DAE+CQL이 CQL보다 나음 |

`summarize.py`가 결과를 모아 표, 그림, 항목별 판정(지지 / 반증 / 판단 보류)을 만든다.

## 비교하는 정책 세 가지

| 프리셋 | 내용 |
| --- | --- |
| `cql` | AD4RL 원본 DDPG+CQL |
| `dae_orig` | 지금까지 실험한 DAE+CQL 그대로. λ=1.0, 학습 중 state에 노이즈 0.2를 넣고 오차 계산, 현재 state 기준, warmup 10000 |
| `dae_fix` | 의도에 맞게 고친 DAE+CQL. 노이즈 없이 오차 계산, 행동의 결과인 next_state 기준, λ = k × (버퍼 reward 표준편차), 기본 k=1.0 |

`dae_fix`의 k=1.0은 출발점일 뿐이다. 바꿔보려면 `--lam-k 0.5 --tag k0.5`처럼 꼬리표를 붙여 따로 저장한다.

## 평가 조건

| 조건 | 내용 |
| --- | --- |
| `clean` | 학습한 시나리오, 입력 그대로 |
| `noise` | 19개 특징에 각각 독립 가우시안 노이즈 (σ=0.2) |
| `noise_orig` | 기존 평가 코드와 같은 노이즈 (값 하나를 뽑아 19개 특징에 똑같이 더함) |
| `denoise` | `noise`와 같은 노이즈를 넣은 뒤 DAE로 복원해서 정책에 넣음 (입력 정화 대안) |
| `lanereduction` | 학습 때 못 본 lane reduction 시나리오(MA_4BL) |
| `cutin` | 학습 때 못 본 cut-in 시나리오(UnifiedRing) |

세 정책은 한 프로세스 안에서 같은 SUMO 시드, 같은 에피소드 시드, 같은 노이즈로 평가된다.
그래서 정책 간 차이를 "같은 상황에서 누가 더 잘했나"로 비교할 수 있다(짝지은 비교).

## 실행

서버에서 저장소 루트(`~/AE_OfflineRL`)에 `verify/` 폴더가 있으면 된다.

### 1. 먼저 짧게 시험 (10~20분)

서버 환경에서 SUMO·flow까지 제대로 도는지만 확인한다.

```bash
cd ~/AE_OfflineRL
DATASETS="highway-NGSIM" SEEDS="5" EPISODES=1 \
V1_ARGS="--dae-epochs 3" TRAIN_ARGS="--epochs 1 --itr 2000" \
bash verify/run_all.sh all
```

끝나고 `verify_out/results/summary.md`가 생기면 성공이다. 오류가 나면 `verify_out/logs/`의 해당 로그를 확인한다.
시험 결과는 지우고 본 실행을 한다.

```bash
rm -rf verify_out
```

### 2. 본 실행

```bash
nohup bash verify/run_all.sh all > verify_out_run.log 2>&1 &
tail -f verify_out_run.log
```

기본 설정은 데이터셋 3개(NGSIM, final-medium, humanlike) × seed 3개(5, 6, 7) × 정책 3개다.
바꾸려면 앞에 변수를 붙인다.

```bash
DATASETS="highway-NGSIM highway-final-medium" SEEDS="5 6 7 8 9" GPUS="0 1 2 3" \
nohup bash verify/run_all.sh all > verify_out_run.log 2>&1 &
```

단계별로 따로 돌릴 수도 있다: `bash verify/run_all.sh 1` (또는 2, 3, 4).
2단계는 이미 학습된 정책이 있으면 건너뛰니, 중간에 끊겨도 다시 실행하면 이어서 한다.

### 걸리는 시간

- 1단계(DAE): 데이터셋·seed당 수십 분 이내.
- 2단계(학습): 원래 코드는 15,000 step마다 SUMO 평가를 해서 오래 걸렸다. 여기서는 평가 없이 학습만 하므로
  훨씬 짧다. 정확한 시간은 로그의 `남은 예상`에 나온다.
- 3단계(평가): 에피소드마다 SUMO를 다시 띄운다. 에피소드 하나에 걸린 시간은 `eval_episodes.csv`의 `wall_sec`에 기록된다.
  1번 짧은 시험에서 이 값을 보고 전체 시간을 계산하면 된다.
  (조건 6개 × 에피소드 5개 × 정책 3개 = 한 데이터셋·seed당 90 에피소드)

## 결과 파일

```
verify_out/
  results/
    summary.md               표와 판정 요약 (보고서에 바로 쓸 수 있음)
    summary_eval.png         조건별 성능·조기 종료·방문 state 오차 그림
    v1_hist_*.png            정상 vs OOD 재구성오차 분포
    v1_parts/*.csv           검증 1 원자료
    eval_episodes.csv        검증 2·3 원자료 (에피소드마다 한 줄)
    train_logs/*.csv         학습 손실, 패널티 평균·적용 비율
  checkpoints/
    dae/                     측정용 DAE (dae_fix 학습에도 같은 것을 씀)
    policy/                  학습된 정책
  logs/                      각 실행의 표준 출력
```

`summary.md`에서 CQL 대비 차이에 `*`가 붙은 값은 짝지은 차이의 평균 ± 2×표준오차가 0을 벗어난 경우다.

## 코드를 점검하며 발견한 것 (기존 결과 해석에 영향)

1. **기존 평가 노이즈는 특징별 노이즈가 아니었다.** 관측 딕셔너리에는 에이전트가 하나라서
   `np.random.randn(len(state_values))`는 값 하나를 만든다. 매 step 같은 값이 19개 특징 전체에 더해졌다.
   `noise_orig` 조건이 이 방식을 재현하고, `noise` 조건이 원래 의도한 특징별 노이즈다.
2. **학습 중 패널티가 노이즈에 좌우됐을 수 있다.** `DDPGCQL_DAE._compute_shaped_reward`는 버퍼 state에 매번
   새 노이즈(σ=0.2)를 넣고 오차를 계산한다. 같은 state라도 뽑힌 노이즈에 따라 패널티가 달라진다.
   검증 1b의 "반복 상관"이 이 정도를 잰다.
3. **같은 seed여도 SUMO 교통 흐름은 매번 달랐다.** 시나리오 설정 파일이 import될 때 `random.randrange`로
   SUMO 시드를 정하는데, 기존 `set_seed()`는 numpy와 torch만 고정한다. 같은 seed 반복 실행의 큰 편차가
   여기서 온다. `evaluate.py`는 이 시드를 고정한다.
4. **warmup은 켜져 있었다.** `main_DDPGCQL_DAE_v2.py`의 `--shape-warmup-steps` 기본값은 10000이다
   (450,000 step 중 처음 약 2%).
5. **DAE의 validation은 실제로 분리돼 있지 않았다.** `pretrain_dae`는 train과 val을 같은 버퍼에서 무작위로
   뽑는다. `v1_dae_ood.py`는 10%를 끝까지 안 보이게 떼어내고 평가한다.
6. **collision_count는 항상 0이었다.** 기존 로그가 모두 `collisions: 0.00`이다. 환경의 info에 `collision`
   키가 없는 것으로 보인다. `evaluate.py`는 사고 페널티(reward ≤ −4.5, 설정 파일의 `accident_penalty = −5`)와
   조기 종료로 사고를 판정한다.

## 가정

- 시나리오 대응: `MA_5LC` = highway, `MA_4BL` = lane reduction, `UnifiedRing` = cut-in.
  세 설정 모두 같은 환경 클래스(`MADRingLCPOEnv`)와 19차원 관측을 쓴다.
- 조기 종료: 에피소드가 horizon(3000 step)에 닿기 전에 끝나면 조기 종료로 본다.
- 측정용 DAE는 학습 데이터의 90%로 학습한다. 원래 실험의 DAE(100%)와 약간 다르다.
- 원래 평가의 `correction_reward` 계산식(`reward + timestep − warmup`)을 그대로 따랐다.
  timestep을 못 읽으면 `warmup + step 수`로 대신한다.
