# 하이퍼파라미터 제안 (PPO, VMAS)

## 시나리오별 추천 설정

### Navigation (3 agents)
```bash
python -m scripts.train \
--scenario-name navigation \
--n-agents 3 \
--frames-per-batch 6000 \
--max-steps 100 \
--n-iters 10 \
--num-epochs 30 \
--minibatch-size 400 \
--lr 3e-4 \
--max-grad-norm 1.0 \
--clip-epsilon 0.2 \
--entropy-eps 1e-4 \
--gamma 0.99 \
--lmbda 0.9
```

### Reverse transport (3 agents)
옮기는 속도가 느리므로 `max-steps` 를 늘림.
```bash
python3 -m scripts.train \
  --scenario-name reverse_transport \
  --n-agents 3 \
  --frames-per-batch 32768 \
  --max-steps 500 \
  --n-iters 250 \
  --num-epochs 4 \
  --minibatch-size 8192 \
  --lr 3e-4 \
  --entropy-eps 1e-4
```

### Wheel (4 agents)
환경이 까다로워서 **수집량과 배치 크기를 키워서** 
아직 안정적으로 학습하는 설정을 찾진 못 함.

```bash
python3 -m scripts.train \
  --scenario-name wheel \
  --n-agents 4 \
  --frames-per-batch 6000 \
  --max-steps 100 \
  --n-iters 500 \
  --num-epochs 4 \
  --minibatch-size 4096 \
  --lr 1e-3 \
  --clip-epsilon 0.2 \
  --entropy-eps 1e-2 \
  --gamma 0.99 \
  --lmbda 0.95
```

### Buzz wire
충돌 패널티가 커서 **탐색을 유도해야 하는** 환경이므로 엔트로피 계수를 약간 높게:

```bash
python3 -m scripts.train \
  --scenario-name buzz_wire \
  --frames-per-batch 4096 \
  --max-steps 150 \
  --n-iters 20 \
  --num-epochs 3 \
  --minibatch-size 1024 \
  --lr 5e-4 \
  --entropy-eps 5e-4
```

## 공통 팁

- **더 길게 학습하고 싶을 때**  
  `--n-iters`, `--frames-per-batch` 를 늘리고, `--num-epochs` 는 3–5 사이에서 조절.
- **미니배치 크기**  
  `--minibatch-size` 는 보통 `frames-per-batch` 의 1/4 ~ 1/6 정도가 안정적.
- **CUDA 사용**  
  GPU 사용 시 `--device cuda:0` 지정. 메모리 부족하면 `frames-per-batch` 와 `minibatch-size` 를 줄이기.
- **TensorBoard 로그 확인**  
  ```bash
  tensorboard --logdir output/<scenario_name>
  ```
  예: `output/navigation`, `output/wheel` 등.


---

- max_steps: 에피소드 길이 상한. VMAS에서 한 에피소드에 허용되는 스텝 수이며, frames_per_batch / max_steps로 동시에 돌리는 env 수가 결정됩니다. 값을 키우면 한 에피소드가 더 길어집니다.
- n_iters: 전체 학습에서 “배치 수집→PPO 업데이트”를 몇 번 반복할지. 한 iter마다 frames_per_batch만큼 샘플을 모으고, 그 샘플로 여러 번 업데이트를 돌립니다.
- num_epochs: 한 iter에서 모은 배치를 몇 번 반복 사용해 PPO 손실을 최적화할지. 같은 데이터를 num_epochs만큼 여러 미니배치로 다시 학습합니다.


---

# 액션 NaN 발산 이슈

## 1. 데이터세트 배치에서 NaN이 발생하는지 확인.

```bash
python3 - <<'PY'
import torch
from scripts.train import build_collector, make_env
from model.policy import build_policy
from src.config import Config

cfg = Config(
    device='cpu',          # GPU 문제 피하려면 cpu
    scenario_name='navigation',
    frames_per_batch=6000,
    max_steps=100,
    n_agents=3,
)

env = make_env(cfg)
policy = build_policy(env, cfg)
collector = build_collector(env, policy, cfg)

try:
    td = next(iter(collector))  # 첫 rollout 배치
    act = td.get(env.action_key)
    obs = td.get(("agents", "observation"))

    def stats(name, x):
        finite = torch.isfinite(x)
        print(f"{name}: finite={finite.all().item()} min={x.min().item()} max={x.max().item()}")

    stats("action", act)
    stats("observation", obs)
finally:
    collector.shutdown()
PY
```

---

## 튜토리얼과 다른 점 찾아 고치기

#### 1. RewardSum

- `RewardSum` 을 추가해도 `episode_reward` 키는 **env 스펙 출력에는 바로 안 보일 수 있다.**
- 그래도 `TransformedEnv` 내부에서는 보상이 누적되어  
  `("agents", "episode_reward")` 키로 **에피소드 누적 보상**이 기록된다.
- 이 상태에서 학습을 돌리면, 튜토리얼과 마찬가지로 **에피소드 리워드의 합(sum)** 을 기준으로 PPO가 학습된다.

정리하면:

- 학습 루프 자체는 **처음부터 튜토리얼과 동일한 PPO 루프였고, 잘 돌아가고 있었다.**
- 문제는 내가 **성능 평가를 reward mean(스텝 평균)** 으로 보면서
  - 코드/튜토리얼 쪽은 **episode reward sum(에피소드 누적 합)** 을 쓰고 있었고,
  - 서로 다른 값을 비교하면서 “안 된다”고 오해하고 삽질하고 있었던 것.
- 결국, 성능 기준은 **reward mean이 아니라 reward sum** 이어야 했고,
  - VMAS 쪽에서 애초에 “에이전트/스텝 보상이 아니라 에피소드 누적 보상”을 기본으로 줬다면
    이렇게 헷갈릴 일이 적었을 문제였다.

현재는:

- `RewardSum` 으로 누적된 보상과 스텝 평균 보상을 **명확히 구분해서 로깅**한다.
  - 누적 보상이 있으면: `episode_reward_sum`  
    - 진행바 / TensorBoard: `train/episode_reward_sum`
  - 누적 보상이 없으면: `episode_reward_mean`  
    - 진행바 / TensorBoard: `train/episode_reward_mean`
  - 스텝 평균 보상은 필요할 때만 참고용으로 쓰고,  
    **성능 평가는 일관되게 “에피소드 보상 합(sum)” 기준으로 본다.**
- 잘못된 `ep_rew` 참조를 제거해서 `NameError` 도 해결했다.

---

#### 2. ("next","done") / ("next","terminated") 를 에이전트 축으로 확장

- 현재 형태:
  - `("next", "done")`, `("next", "terminated")` 의 shape: **[batch, time, 1]**
  - 보상 `("next", *env.reward_key)` 의 shape: **[batch, time, n_agents, 1]**
- “에이전트 축으로 확장”이란:
  - `done` 텐서를 **에이전트 수(`n_agents`)만큼 복제**해서  
    보상과 같은 shape 으로 맞추는 것.
  - 예: `n_agents = 3` 이면  
    `[B, T, 1] → [B, T, 3, 1]`

이렇게 하는 이유:

- VMAS `navigation` 에서는 `done` / `terminated` 가 **환경 전체에 공통**인 플래그라 기본 shape 이 `[B, T, 1]` 이다.
- 하지만 GAE / 손실 계산에서는 보상과 같은 shape (`[B, T, n_agents, 1]`) 를 기대하므로,
  - 같은 값을 에이전트 축으로 복제해서  
    `("next", "agents", "done")`, `("next", "agents", "terminated")` 에 넣는다.

정리하면:

- **에이전트마다 값이 다르지 않다.**  
- 다만, GAE·loss 에서 쓰기 위해 **보상 텐서와 shape 을 맞춰 주기 위해 복제**하는 것이다.

---
