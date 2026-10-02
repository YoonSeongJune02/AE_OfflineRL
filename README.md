# AE_OfflineRL

자율주행 오프라인 강화학습 벤치마크 [AD4RL](https://sites.google.com/view/ad4rl/%ED%99%88) (ICRA 2024) 위에서 진행한 졸업작품입니다.
AD4RL의 DDPG+CQL에 Denoising Autoencoder(DAE)를 붙여, 데이터에서 보기 드문 상태로 가는 것을 보상으로 억제해 봤습니다.

## 아이디어

CQL은 데이터에 없는 **행동**의 Q값을 낮춰서 정책을 보수적으로 만듭니다. 하지만 정책이 어떤 **상태**로 가게 되는지는 따로 보지 않습니다.
이 프로젝트에서는 오프라인 데이터의 state로 DAE를 학습시키고, 재구성오차가 큰 state를 "데이터에서 보기 드문 상태"로 봤습니다.
학습할 때 그런 state에서는 보상을 깎습니다.

```
shaped_reward = reward − λ_t · penalty

z       = (재구성오차 − 이동평균) / 이동표준편차
penalty = tanh(max(0, z − 1))
λ_t     = λ_max · min(1, t / 10000)      # 처음 10,000 step 동안 0에서 λ_max까지 올림
```

DAE는 CQL 학습 전에 버퍼 state로 먼저 학습하고, CQL 학습 중에는 고정해서 씁니다.

## 파일 구성

| 파일 | 내용 |
| --- | --- |
| `main_DDPGCQL.py` | Baseline. AD4RL의 DDPG+CQL |
| `main_DDPGCQL_DAE_v2.py` | 제안 방법. DAE 사전학습 → 보상 조정을 넣은 DDPG+CQL 학습 |
| `Algos/DDPG_CQL.py` | DDPG+CQL |
| `Algos/DDPG_CQL_DAE_v2.py` | DDPG+CQL에 보상 조정을 더한 버전 |
| `Algos/DAE_v2.py` | DAE 모델과 사전학습 함수 |
| `Algos/reward_shaping_v2.py` | 재구성오차 → 패널티 변환 (`RewardShaper`) |
| `run_experiments.sh` | 데이터셋 × seed 조합으로 학습 실행 |
| `verify/` | 제안 방법이 의도대로 동작하는지 확인하는 검증 코드 ([설명](verify/README.md)) |
| `tests/test_dae_v2.py` | DAE·RewardShaper 단위 테스트 |

나머지 `main_*.py`(BC, BCQ, AWAC, EDAC, PLAS, DDPGBC)와 `exp_configs/`, `requirements/`는 AD4RL 원본입니다.

## 실행

환경 설치는 아래 AD4RL 원본 안내를 따릅니다. 데이터셋은 `buffers/<데이터셋 이름>/`에 `state.npy`, `action.npy`, `next_state.npy`, `reward.npy`, `done.npy`로 둡니다.

```bash
# Baseline (CQL)
ALGO=cql GPU=<번호> bash run_experiments.sh

# 제안 방법 (DAE+CQL), λ_max = 1.0
ALGO=dae GPU=<번호> EXTRA="--shape-lambda-max 1.0" bash run_experiments.sh

# 데이터셋과 seed 지정
ALGO=cql GPU=<번호> DATASETS="highway-NGSIM highway-humanlike" SEEDS="5 6 7" bash run_experiments.sh
```

GPU 번호는 실행할 때마다 서버 상황(`nvidia-smi`)을 보고 정합니다. 지정하지 않으면 실행되지 않습니다.

결과는 WandB에 `CQL_<데이터셋>_seed<번호>`, `DAE_CQL_<데이터셋>_seed<번호>` 이름으로 기록됩니다.
학습 중 평가 점수는 AD4RL 논문과 같은 `correction_reward`입니다.

검증 실험은 [verify/README.md](verify/README.md)를 보세요.

```bash
python tests/test_dae_v2.py
```

---

아래는 AD4RL 원본 README입니다.

# AD4RL (원본)

[//]: # (<div>)

[//]: # (    <img src="https://img.shields.io/badge/Ubuntu-E95420?style=flat-square&logo=Ubuntu&logoColor=white"/> )

[//]: # (    <img src="https://img.shields.io/github/languages/top/leexim/AD4RL"> )

[//]: # (    <img src="https://img.shields.io/github/languages/code-size/leexim/AD4RL"> )

[//]: # (    <img src="https://img.shields.io/github/license/LeeXim/AD4RL">)

[//]: # (    <img src="https://img.shields.io/github/v/release/LeeXim/AD4RL">)

[//]: # (  </div>)

[//]: # (  <br>)

## ICRA2024
#### AD4RL: Autonomous Driving based on Dataset-driven Deep Reinforcement Learning 
[[Webpage]](https://sites.google.com/view/ad4rl/%ED%99%88) 
[[Dataset]](https://drive.google.com/drive/folders/1OKUS8rqLZ_REk1SP8PyAVYE_MMwlSVvo?usp=sharing)


## 1. FLOW Framework
See https://flow-project.github.io/ for Detail information of this framework

### Installation
#### A. Anaconda with Python 3
1. Install prequisites: `sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6` 
2. Download the Anaconda installation file for Linux in [Anaconda](https://anaconda.com/), and unzip the file
3. Install Anaconda `bash ~/Downloads/Anaconda3-2023.03-1-Linux-x86_64.sh`
> **_NOTE:_**  we recommend you to running conda init '**yes**'.

#### B. FLOW installation
Following the below scripts in your terminal.
```
# Download FLOW github repo'.
git clone https://github.com/flow-project/flow.git
cd flow

# Create a conda env and install the FLOW
conda env create -f environment.yml
conda activate flow
python setup.py develop

# install flow on previoulsy created environment 
pip install -e .
```

###### B-1. SUMO installation
Install driving simulator (SUMO) 
```
bash scripts/setup_sumo_ubuntu1804.sh
which sumo
sumo --version
sumo-gui
```
Testing the connection between FLOW and SUMO
```
conda activate flow
python examples/simulate.py ring
```

###### B-2. Pytorch installation
Install torch: `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`
> **_NOTE:_**  Should install at least 1.6.0 version of pytorch (Recommend torch = 1.11.0 & cudatoolkit=10.2).\
> Check the [Pytorch Documents](https://pytorch.org/get-started/previous-versions/).

###### B-3. Ray RLlib installation
Install Ray: `pip install -U ray==0.8.7`
> **_NOTE:_**  Should install at least 0.8.6 version of Ray. (Recommend 0.8.7).

###### B-4. Python Library installation
```

```

## 2. AD4RL
### Repository
Clone this library: `git clone` (Will be updated)
### Dependencies Update
```
sh ./requirements/env_requirements.sh
```
### Driving Scenarios
We provide the three driving scenarios as following table.
- Click Driving Scenario, You can check the illustrative image about driving scenario
- Click exp_config, You can check the code about driving scenario

| Driving Scenario                                   | exp_config                                                    |
|----------------------------------------------------|---------------------------------------------------------------|
| [Cut-in](exp_configs%2FFig%2Fcutin_Part.pdf)       | [UnifiedRing](exp_configs%2Frl%2Fmultiagent%2FUnifiedRing.py) |
| [Lane Reduction](exp_configs%2FFig%2FLR_Part.pdf)  | [MA_4BL](exp_configs%2Frl%2Fmultiagent%2FMA_4BL.py)           |
| [Highway](exp_configs%2FFig%2FHigh_Part.pdf)       | [MA_5LC](exp_configs%2Frl%2Fmultiagent%2FMA_5LC.py)           |

### Dataset
You can access the AD4RL googledrive by clicking the title name of driving scenario. 

| [Cut-in](https://drive.google.com/drive/folders/1qWV2UugbpeENPEAxONr0DRGnh9fiXJJS?usp=sharing)    | [Lane Reduction](https://drive.google.com/drive/folders/1vUeZEKXBO2_TC9LdxkH6JS4rMootnMfb) | [Highway](https://drive.google.com/drive/folders/1ZPxCGrZGrIYkU6VsKUYGWeMSPGdZOWcH?usp=sharing) |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------| 
| cutin-expert                                                                                      | lanereduction-expert                                                                       | highway-expert                                                                                  | 
| cutin-medium                                                                                      | lanereduction-medium                                                                       | highway-medium                                                                                  | 
| cutin-random                                                                                      | lanereduction-random                                                                       | highway-random                                                                                  |
| cutin-expert-medium                                                                               | lanereduction-expert-medium                                                                | highway-expert-meidum                                                                           |
| cutin-expert-random                                                                               | lanereduction-expert-random                                                                | highway-expert-random                                                                           |
| cutin-humanlike                                                                                   | lanereduction-humanlike                                                                    | highway-humanlike                                                                               |

### Train
 ```
 python [algorithm] [exp_config] --dataset [dataset]
 ```
- **[algorithm]**: main_BC.py, main_BCQ.py, main_DDPGBC.py, main_EDAC.py, main_PLAS.py
- **[exp_config]**: UnifiedRing, MA_4BL, MA_5LC
- **[dataset]**: See the above table (e.g., cutin-expert, highway-NGSIM)
