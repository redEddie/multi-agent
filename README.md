# Multi-Agent PPO with TorchRL (Navigation and More)

이 프로젝트는 PyTorch TorchRL의 MARL 튜토리얼을 바탕으로 시작했지만, 실제 학습이 불안정하거나 이해하기 어려운 부분을 직접 고쳐보며 정리한 코드입니다. `navigation`, `wheel`, `reverse_transport`, `buzz_wire` 등 VMAS 시나리오를 동일한 학습 루프로 돌릴 수 있고, 학습된 정책을 ONNX로 내보내어 재생(GIF 포함)할 수 있습니다.

<hr style="border: 2px solid #000; margin: 24px 0;">

## 환경 정보
- Python: 3.10 (conda 환경 `marl` 기준)
- 주요 패키지(검증 버전): torch==2.9.1, torchrl==0.10.1, vmas==1.5.2, tqdm==4.67.1, onnxruntime==1.23.2, imageio==2.37.2

## 새 conda 환경 생성
### 1) 최소 예제용(가장 간단)
```bash
conda create -n marl python=3.10
conda activate marl
pip install torch torchrl vmas tqdm onnxruntime imageio
```

### 2) environment.yml 로 동일 환경 재현
```bash
conda env create -f environment.yml
conda activate marl
```

### 3) requirements.txt 로 설치
```bash
conda create -n marl python=3.10
conda activate marl
pip install -r requirements.txt
```


## 기존 conda 환경에 패키지만 추가하기

### 1) 기본 설치
```bash
conda activate <your_env>
pip install torch torchrl vmas tqdm onnxruntime imageio
```

### 2) 검증된 버전 세트 설치(requirements 사용)
```bash
conda activate <your_env>
pip install -r requirements.txt
```

<hr style="border: 2px solid #000; margin: 24px 0;">

## 학습/재생
- 학습 예시(튜토리얼 기본 하이퍼):
```bash
python -m scripts.train --scenario-name navigation --n-agents 3 \
  --frames-per-batch 6000 --max-steps 100 --n-iters 10 --num-epochs 30 \
  --minibatch-size 400 --lr 3e-4 --max-grad-norm 1.0 \
  --clip-epsilon 0.2 --entropy-eps 1e-4 --gamma 0.99 --lmbda 0.9 \
  --device cuda:0
```
- 재생(ONNX 우선, GIF 자동 저장):
```bash
python -m scripts.play --scenario-name navigation --n-steps 200 --num-envs 1 --device cuda:0
```
최신 run이 자동으로 선택되며, `output/<scenario>/<timestamp>/rollout.gif`가 생성됩니다.
