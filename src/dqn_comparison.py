"""DQN 비교: 기존 DDM vs 개선 DDM (PGD ε=0.03 공격 환경).

FRDDM-DQN 논문 기반 2D 드론 환경에서 두 가지 버전의 DQN을 비교한다.

버전 A (기존 DDM):
  - 모든 경험을 리플레이 메모리에 저장
  - PGD ε=0.03 공격: D_obstacle 0.39% 오차, theta_obstacle 0.38도 오차
  - 1000 에피소드 학습

버전 B (개선 DDM):
  - shift > 10px 경험은 CE로 분류하여 저장 차단
  - RISK 상태(장애물 2× 반경 이내)에서 보상 -0.5 페널티 추가
  - PGD ε=0.03 공격 동일하게 적용
  - 2000 에피소드 학습 (CE 필터링으로 줄어드는 유효 경험 보정)

환경:
  - 비행 공간: 15km × 18km 2D 평면
  - 드론 속도: 0.3km/스텝
  - 장애물: 5개 고정, 반경 1.5km
  - 최대 스텝: 200/에피소드

상태 벡터 (논문 수식 18, 19 기반):
  [D_goal, theta_goal, D_obstacle, theta_obstacle]
  - theta: 드론 헤딩 기준 상대 방위각 (정규화)

행동 공간:
  0=직진, 1=좌15°, 2=우15°, 3=좌30°, 4=우30°

보상 함수:
  목적지 도달 +1 / 충돌 -1 / 이탈 -1 / 그 외 0
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────
_prefer = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic',
           'Gulim', 'Dotum', 'Batang', 'HCR Batang']
_available = {f.name for f in fm.fontManager.ttflist}
for _font in _prefer:
    if _font in _available:
        plt.rcParams['font.family'] = _font
        break
plt.rcParams['axes.unicode_minus'] = False

# ── 재현성 ──────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 경로 ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 환경 설정
# ══════════════════════════════════════════════════════════════════════════════
SPACE_W    = 15.0   # km
SPACE_H    = 18.0   # km
STEP_SIZE  = 0.3    # km/스텝
GOAL_RADIUS = 1.0   # km – 도착 판정 반경
OBS_RADIUS  = 1.5   # km – 장애물 충돌 반경
MAX_STEPS   = 200   # 에피소드당 최대 스텝

DIAG = math.sqrt(SPACE_W ** 2 + SPACE_H ** 2)  # ≈ 23.43 km (정규화 기준)

# 5개 고정 장애물 좌표 (km)
OBSTACLES: list[tuple[float, float]] = [
    (3.0,  3.0),
    (7.5,  9.0),
    (12.0, 6.0),
    (5.0, 15.0),
    (10.0, 12.0),
]

# ══════════════════════════════════════════════════════════════════════════════
# PGD ε=0.03 공격 파라미터 (analyze.py 결과 기반)
# ══════════════════════════════════════════════════════════════════════════════
PGD_MEAN_SHIFT   = 13.8   # px  – ε=0.03 평균 bbox 중심 오차
PGD_STD_SHIFT    = 4.0    # px  – 표준편차 (가정)
PGD_D_ERR_RATE   = 0.0039  # D_obstacle 오차율 (0.39%)
PGD_T_ERR_DEG    = 0.38   # theta_obstacle 오차 (도)
TAU              = 0.05   # 픽셀→월드 변환 계수 (analyze.py 기준)
CE_SHIFT_THRESH  = 10.0   # px  – CE 판정 임계값 (improved_ddm.py 기준)

# RISK 판정: 장애물 반경 2× 이내
RISK_DIST = OBS_RADIUS * 2.0  # 3.0 km

# ══════════════════════════════════════════════════════════════════════════════
# DQN 하이퍼파라미터
# ══════════════════════════════════════════════════════════════════════════════
STATE_DIM          = 4
ACTION_DIM         = 5
HIDDEN_DIM         = 64
MEMORY_CAPACITY    = 10_000
BATCH_SIZE         = 32
GAMMA              = 0.95
LR                 = 0.001
TARGET_UPDATE_FREQ = 100    # 학습 스텝마다 타겟 네트워크 동기화
EPSILON_START      = 1.0
EPSILON_END        = 0.05
EPSILON_DECAY      = 0.995  # 에피소드별 감쇠
EPISODES           = 1000   # 기본값 (버전 A)
EPISODES_A         = 1000   # 기존 DDM 에피소드 수
EPISODES_B         = 2000   # 개선 DDM 에피소드 수 (CE 필터링 보정)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 드론 환경
# ══════════════════════════════════════════════════════════════════════════════

class DroneEnv:
    """15km × 18km 2D 드론 비행 환경.

    상태 벡터 (논문 수식 18, 19):
      [D_goal_norm, theta_goal_norm, D_obs_norm, theta_obs_norm]
      - D: 거리 / DIAG  → [0, 1]
      - theta: 드론 헤딩 기준 상대 방위각 / 180  → [-1, 1]
    """

    def __init__(self) -> None:
        self.pos: list[float] = [0.0, 0.0]
        self.heading: float   = 0.0      # 도
        self.goal: list[float] = [0.0, 0.0]
        self.steps: int       = 0

    # ── 초기화 ────────────────────────────────────────────────────────────────

    def reset(self) -> list[float]:
        """드론과 목적지를 랜덤 초기화 후 초기 상태 반환."""
        self.pos     = self._random_free_pos()
        self.heading = random.uniform(0.0, 360.0)
        self.goal    = self._random_free_pos(min_dist_from=self.pos, min_dist=3.0)
        self.steps   = 0
        return self._state()

    def _random_free_pos(
        self,
        min_dist_from: list[float] | None = None,
        min_dist: float = 0.0,
        max_tries: int = 1000,
    ) -> list[float]:
        """장애물과 OBS_RADIUS*2 이상 떨어진 랜덤 위치 반환."""
        for _ in range(max_tries):
            x = random.uniform(0.5, SPACE_W - 0.5)
            y = random.uniform(0.5, SPACE_H - 0.5)
            if self._min_obs_dist(x, y) <= OBS_RADIUS * 2:
                continue
            if min_dist_from is not None:
                if math.dist([x, y], min_dist_from) < min_dist:
                    continue
            return [x, y]
        # fallback
        return [SPACE_W / 2, SPACE_H / 2]

    # ── 내부 유틸 ─────────────────────────────────────────────────────────────

    def _min_obs_dist(self, x: float, y: float) -> float:
        return min(math.dist([x, y], list(o)) for o in OBSTACLES)

    def _nearest_obstacle(self) -> tuple[float, float]:
        """가장 가까운 장애물까지 (거리 km, 상대 방위각 도) 반환.

        논문 수식 18, 19:
          D_obstacle  = Euclidean distance (드론–장애물)
          theta_obstacle = atan2(o_y - d_y, o_x - d_x) - drone_heading
                          → [-180, 180]
        """
        x, y = self.pos
        best_d   = float("inf")
        best_obs = OBSTACLES[0]
        for obs in OBSTACLES:
            d = math.dist([x, y], list(obs))
            if d < best_d:
                best_d, best_obs = d, obs

        ox, oy = best_obs
        abs_angle = math.degrees(math.atan2(oy - y, ox - x))
        rel_angle = (abs_angle - self.heading + 180.0) % 360.0 - 180.0
        return best_d, rel_angle

    def _state(self) -> list[float]:
        x, y  = self.pos
        gx, gy = self.goal

        # 목적지
        D_goal     = math.dist(self.pos, self.goal) / DIAG
        abs_g_ang  = math.degrees(math.atan2(gy - y, gx - x))
        theta_goal = (abs_g_ang - self.heading + 180.0) % 360.0 - 180.0

        # 장애물 (논문 수식 18, 19)
        D_obs, theta_obs = self._nearest_obstacle()

        return [
            D_goal,
            theta_goal / 180.0,
            D_obs / DIAG,
            theta_obs / 180.0,
        ]

    def get_raw_obstacle(self) -> tuple[float, float]:
        """정규화 전 D_obstacle(km), theta_obstacle(도) 반환 – 공격 계산용."""
        return self._nearest_obstacle()

    # ── 스텝 ──────────────────────────────────────────────────────────────────

    def step(
        self, action: int
    ) -> tuple[list[float], float, bool, dict]:
        """행동 실행 후 (next_state, reward, done, info) 반환.

        행동 공간:
          0=직진  1=좌15°  2=우15°  3=좌30°  4=우30°
        """
        delta = {0: 0, 1: 15, 2: -15, 3: 30, 4: -30}[action]
        self.heading = (self.heading + delta) % 360.0

        rad = math.radians(self.heading)
        self.pos[0] += STEP_SIZE * math.cos(rad)
        self.pos[1] += STEP_SIZE * math.sin(rad)
        self.steps  += 1

        x, y = self.pos
        obs_dist     = self._min_obs_dist(x, y)
        goal_reached = math.dist(self.pos, self.goal) < GOAL_RADIUS
        collision    = obs_dist < OBS_RADIUS
        out_of_bounds = not (0.0 <= x <= SPACE_W and 0.0 <= y <= SPACE_H)
        timeout      = self.steps >= MAX_STEPS
        done         = goal_reached or collision or out_of_bounds or timeout

        if goal_reached:
            reward = 1.0
        elif collision or out_of_bounds:
            reward = -1.0
        else:
            reward = 0.0

        info = {
            "goal_reached":  goal_reached,
            "collision":     collision,
            "out_of_bounds": out_of_bounds,
            "obs_dist":      obs_dist,
        }
        return self._state(), reward, done, info


# ══════════════════════════════════════════════════════════════════════════════
# 2. PGD ε=0.03 공격 시뮬레이터
# ══════════════════════════════════════════════════════════════════════════════

def pgd_attack(
    state: list[float],
    D_obs_km: float,
    theta_obs_deg: float,
) -> tuple[list[float], float]:
    """PGD ε=0.03 공격을 상태 벡터에 적용한다.

    analyze.py 결과:
      ε=0.03: D_obstacle 오차율 0.39%, theta_obstacle 오차 0.38도,
              평균 bbox 중심 shift 13.8px

    shift를 가우시안으로 샘플링하고, 그에 비례하는 오차를 D_obs/theta_obs에 주입.
    등가 픽셀 shift도 함께 반환한다 (CE 판정용).

    Returns:
        corrupted_state: 오염된 4차원 상태 벡터
        shift_px:        등가 bbox 중심 이동량 (픽셀)
    """
    shift = float(abs(np.random.normal(PGD_MEAN_SHIFT, PGD_STD_SHIFT)))
    ratio = shift / PGD_MEAN_SHIFT

    d_err   = PGD_D_ERR_RATE * ratio * random.choice([-1, 1])
    t_err   = PGD_T_ERR_DEG  * ratio * random.choice([-1, 1])

    D_obs_c     = D_obs_km * (1.0 + d_err)
    theta_obs_c = theta_obs_deg + t_err

    corrupted = [
        state[0],
        state[1],
        D_obs_c / DIAG,
        theta_obs_c / 180.0,
    ]
    return corrupted, shift


# ══════════════════════════════════════════════════════════════════════════════
# 3. Q-네트워크
# ══════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """4 → 64 → 64 → 5 예측/타겟 네트워크."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, ACTION_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 리플레이 메모리
# ══════════════════════════════════════════════════════════════════════════════

Transition = tuple  # (state, action, reward, next_state, done)


class ReplayMemory:
    """원형 리플레이 버퍼 (버전 A용)."""

    def __init__(self, capacity: int = MEMORY_CAPACITY) -> None:
        self.buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buf.append(args)

    def sample(self, n: int) -> list[Transition]:
        return random.sample(self.buf, n)

    def __len__(self) -> int:
        return len(self.buf)


# ══════════════════════════════════════════════════════════════════════════════
# 5. DQN 에이전트
# ══════════════════════════════════════════════════════════════════════════════

class DQNAgent:
    """예측 네트워크 + 타겟 네트워크 DQN."""

    def __init__(self) -> None:
        self.policy_net = QNetwork()
        self.target_net = QNetwork()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer    = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory       = ReplayMemory()
        self.epsilon      = EPSILON_START
        self.train_steps  = 0   # 실제 gradient update 횟수

    def act(self, state: list[float]) -> int:
        if random.random() < self.epsilon:
            return random.randrange(ACTION_DIM)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            return int(self.policy_net(s).argmax().item())

    def push(self, state, action, reward, next_state, done) -> None:
        self.memory.push(state, action, reward, next_state, float(done))

    def update(self) -> float | None:
        if len(self.memory) < BATCH_SIZE:
            return None

        batch  = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(states)
        actions     = torch.LongTensor(actions).unsqueeze(1)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones       = torch.FloatTensor(dones)

        q_vals  = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q  = self.target_net(next_states).max(1)[0]
            targets = rewards + GAMMA * next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


# ══════════════════════════════════════════════════════════════════════════════
# 6. 버전 A – 기존 DDM (모든 경험 저장 + PGD 공격)
# ══════════════════════════════════════════════════════════════════════════════

def train_version_a() -> tuple[list[float], list[float], dict]:
    """버전 A 학습 (EPISODES_A=1000).

    - PGD ε=0.03 공격으로 상태 벡터 오염
    - 오염 여부와 무관하게 모든 경험 리플레이 메모리에 저장

    Returns:
        arr_hist:  에피소드별 최근 100ep 도착률
        col_hist:  에피소드별 최근 100ep 충돌률
        stats:     {total_steps, ce_blocked, valid_exp,
                    valid_exp_hist, arrival_count, episodes}
    """
    print(f"\n[버전 A] 기존 DDM 학습 시작 (PGD ε=0.03 공격, 모든 경험 저장)"
          f"  [{EPISODES_A}에피소드]")
    env   = DroneEnv()
    agent = DQNAgent()

    arr_hist: list[float] = []
    col_hist: list[float] = []
    window_arr = deque(maxlen=100)
    window_col = deque(maxlen=100)

    total_steps     = 0
    arrival_count   = 0
    valid_exp_hist: list[int] = []   # 에피소드별 누적 유효 경험 수

    for ep in range(EPISODES_A):
        state   = env.reset()
        arrived = False
        hit     = False

        for _ in range(MAX_STEPS):
            D_obs_km, theta_obs_deg = env.get_raw_obstacle()
            c_state, _shift = pgd_attack(state, D_obs_km, theta_obs_deg)

            action = agent.act(c_state)
            next_state, reward, done, info = env.step(action)
            total_steps += 1   # 기존 DDM은 모든 경험이 유효

            D_obs_next, theta_obs_next = env.get_raw_obstacle()
            c_next, _ = pgd_attack(next_state, D_obs_next, theta_obs_next)

            # 모든 경험 저장 (기존 DDM)
            agent.push(c_state, action, reward, c_next, done)
            agent.update()

            state = next_state
            if info["goal_reached"]:
                arrived = True
                break
            if info["collision"] or info["out_of_bounds"] or done:
                hit = info["collision"] or info["out_of_bounds"]
                break

        agent.decay_epsilon()
        if arrived:
            arrival_count += 1
        window_arr.append(1 if arrived else 0)
        window_col.append(1 if hit else 0)
        arr_hist.append(sum(window_arr) / len(window_arr))
        col_hist.append(sum(window_col) / len(window_col))
        valid_exp_hist.append(total_steps)   # 기존 DDM: 누적 유효 = 누적 스텝

        if (ep + 1) % 100 == 0:
            print(f"  Ep {ep+1:4d} | 도착률={arr_hist[-1]:.3f}"
                  f" | 충돌률={col_hist[-1]:.3f}"
                  f" | ε={agent.epsilon:.3f}"
                  f" | 유효경험={total_steps:,}")

    stats = {
        "total_steps":    total_steps,
        "ce_blocked":     0,
        "valid_exp":      total_steps,
        "valid_exp_hist": valid_exp_hist,
        "arrival_count":  arrival_count,
        "episodes":       EPISODES_A,
    }
    return arr_hist, col_hist, stats


# ══════════════════════════════════════════════════════════════════════════════
# 7. 버전 B – 개선 DDM (CE 필터링 + RISK 페널티 + PGD 공격)
# ══════════════════════════════════════════════════════════════════════════════

def _train_improved_ddm(
    ce_threshold: float,
    label: str,
    episodes: int = EPISODES,
    verbose: bool = True,
) -> tuple[list[float], list[float], dict]:
    """개선 DDM 학습 공통 루틴.

    Args:
        ce_threshold: CE 판정 픽셀 임계값 (shift > ce_threshold → CE)
        label:        로그 출력용 이름
        episodes:     학습 에피소드 수
        verbose:      100에피소드마다 중간 결과 출력 여부

    Returns:
        arr_hist:  에피소드별 최근 100ep 도착률
        col_hist:  에피소드별 최근 100ep 충돌률
        stats:     {total_steps, ce_blocked, valid_exp,
                    valid_exp_hist, arrival_count, episodes}
    """
    env   = DroneEnv()
    agent = DQNAgent()

    arr_hist: list[float] = []
    col_hist: list[float] = []
    window_arr = deque(maxlen=100)
    window_col = deque(maxlen=100)

    ce_blocked      = 0
    total_steps     = 0
    valid_exp       = 0
    arrival_count   = 0
    valid_exp_hist: list[int] = []

    for ep in range(episodes):
        state   = env.reset()
        arrived = False
        hit     = False

        for _ in range(MAX_STEPS):
            D_obs_km, theta_obs_deg = env.get_raw_obstacle()
            c_state, shift = pgd_attack(state, D_obs_km, theta_obs_deg)

            action = agent.act(c_state)
            next_state, reward, done, info = env.step(action)
            total_steps += 1

            # RISK 상태 페널티 (improved_ddm.py _PENALTY_RISK 참조)
            if info["obs_dist"] < RISK_DIST and not info["goal_reached"]:
                reward += -0.5

            # CE 판정: shift > ce_threshold → 저장 차단
            if shift > ce_threshold:
                ce_blocked += 1
            else:
                D_obs_next, theta_obs_next = env.get_raw_obstacle()
                c_next, _ = pgd_attack(next_state, D_obs_next, theta_obs_next)
                agent.push(c_state, action, reward, c_next, done)
                valid_exp += 1

            agent.update()

            state = next_state
            if info["goal_reached"]:
                arrived = True
                break
            if info["collision"] or info["out_of_bounds"] or done:
                hit = info["collision"] or info["out_of_bounds"]
                break

        agent.decay_epsilon()
        if arrived:
            arrival_count += 1
        window_arr.append(1 if arrived else 0)
        window_col.append(1 if hit else 0)
        arr_hist.append(sum(window_arr) / len(window_arr))
        col_hist.append(sum(window_col) / len(window_col))
        valid_exp_hist.append(valid_exp)

        if verbose and (ep + 1) % 100 == 0:
            discard_rate = ce_blocked / total_steps if total_steps > 0 else 0.0
            print(f"  Ep {ep+1:4d} | 도착률={arr_hist[-1]:.3f}"
                  f" | 충돌률={col_hist[-1]:.3f}"
                  f" | ε={agent.epsilon:.3f}"
                  f" | CE차단={ce_blocked} ({discard_rate:.1%})"
                  f" | 유효경험={valid_exp:,}")

    stats = {
        "total_steps":    total_steps,
        "ce_blocked":     ce_blocked,
        "valid_exp":      valid_exp,
        "valid_exp_hist": valid_exp_hist,
        "arrival_count":  arrival_count,
        "episodes":       episodes,
    }
    return arr_hist, col_hist, stats


def train_version_b() -> tuple[list[float], list[float], dict]:
    """버전 B 학습 (CE 임계값 10px, EPISODES_B=2000)."""
    print(f"\n[버전 B] 개선 DDM 학습 시작 (CE 임계값={CE_SHIFT_THRESH}px, "
          f"RISK 페널티, {EPISODES_B}에피소드)")
    return _train_improved_ddm(
        ce_threshold=CE_SHIFT_THRESH,
        label="버전B",
        episodes=EPISODES_B,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 8. 임계값 민감도 분석 (CE 임계값 10 / 20 / 30 px)
# ══════════════════════════════════════════════════════════════════════════════

CE_THRESHOLDS = [10, 20, 30]   # 분석할 임계값 목록 (px)
_THRESH_COLORS = ["blue", "green", "orange"]


def threshold_analysis() -> None:
    """CE 임계값별 1000에피소드 학습 및 결과 저장.

    - 임계값 10 / 20 / 30 px 각각 1000에피소드 학습
    - results/threshold_analysis.png 그래프 저장
    - 임계값별 최종 도착률, CE 폐기율 표 출력
    """
    print("\n" + "=" * 60)
    print("임계값 민감도 분석: CE 임계값 10 / 20 / 30 px")
    print("=" * 60)

    results: list[dict] = []

    for thresh, color in zip(CE_THRESHOLDS, _THRESH_COLORS):
        print(f"\n[임계값 {thresh}px] 학습 시작")
        arr_hist, col_hist, stats = _train_improved_ddm(
            ce_threshold=float(thresh),
            label=f"{thresh}px",
            episodes=EPISODES,
        )

        # CE 폐기율 이론치: P(|N(μ,σ)| > thresh) 정규분포 누적으로 계산
        import math as _math
        def _norm_cdf(x: float, mu: float, sigma: float) -> float:
            return 0.5 * (1.0 + _math.erf((x - mu) / (sigma * _math.sqrt(2))))
        discard_rate_theory = float(
            1.0 - _norm_cdf(thresh, PGD_MEAN_SHIFT, PGD_STD_SHIFT)
        )

        results.append({
            "threshold_px":       thresh,
            "color":              color,
            "arrival_history":    arr_hist,
            "collision_history":  col_hist,
            "final_arrival_rate": float(np.mean(arr_hist[-100:])),
            "final_collision_rate": float(np.mean(col_hist[-100:])),
            "ce_blocked_total":   stats["ce_blocked"],
            "discard_rate_theory": discard_rate_theory,
        })

    # ── 표 출력 ──────────────────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print("임계값 민감도 분석 결과 (최종 100에피소드)")
    print(sep)
    print(f"{'임계값(px)':>10} {'도착률':>10} {'충돌률':>10}"
          f" {'CE폐기율(이론)':>16} {'CE차단건':>10}")
    print("-" * 62)
    for r in results:
        print(f"{r['threshold_px']:>10d}"
              f" {r['final_arrival_rate']:>10.4f}"
              f" {r['final_collision_rate']:>10.4f}"
              f" {r['discard_rate_theory']:>16.1%}"
              f" {r['ce_blocked_total']:>10d}")
    print(sep)

    # ── 그래프 ───────────────────────────────────────────────────────────────
    episodes = list(range(1, EPISODES + 1))
    fig, ax  = plt.subplots(figsize=(12, 6))

    for r in results:
        thresh = r["threshold_px"]
        rate   = r["final_arrival_rate"]
        discard = r["discard_rate_theory"]
        ax.plot(
            episodes, r["arrival_history"],
            color=r["color"], linewidth=1.5, alpha=0.9,
            label=f"임계값 {thresh}px  (CE폐기율 {discard:.0%}, 최종 도착률 {rate:.3f})",
        )

    ax.set_xlabel("에피소드", fontsize=12)
    ax.set_ylabel("최근 100에피소드 도착률", fontsize=12)
    ax.set_title("CE 임계값별 학습 성능 비교", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, EPISODES)
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    out_png = RESULTS_DIR / "threshold_analysis.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"\n그래프 저장 → {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. 결과 저장 (v2: 확장 표 + 유효 경험 기준 비교)
# ══════════════════════════════════════════════════════════════════════════════

def save_results(
    arr_a: list[float], col_a: list[float],
    arr_b: list[float], col_b: list[float],
    stats_a: dict,
    stats_b: dict,
) -> None:
    """확장 표·그래프·JSON 저장 → results/dqn_comparison_v2.png"""

    final_arr_a = float(np.mean(arr_a[-100:]))
    final_col_a = float(np.mean(col_a[-100:]))
    final_arr_b = float(np.mean(arr_b[-100:]))
    final_col_b = float(np.mean(col_b[-100:]))

    # ── 유효 경험 기준 도착률 계산 ─────────────────────────────────────────
    # 기존 DDM의 유효 경험 수(M)와 동일한 시점의 개선 DDM 도착률
    target_valid = stats_a["valid_exp"]
    match_idx    = next(
        (i for i, v in enumerate(stats_b["valid_exp_hist"]) if v >= target_valid),
        len(arr_b) - 1,
    )
    rate_b_at_match = arr_b[match_idx]   # 개선 DDM이 M개 유효 경험 시점의 도착률
    ep_b_at_match   = match_idx + 1      # 해당 에피소드 번호

    ce_discard_rate = (stats_b["ce_blocked"] / stats_b["total_steps"]
                       if stats_b["total_steps"] > 0 else 0.0)

    # ── 확장 표 출력 ──────────────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print("최종 학습 결과 비교")
    print(sep)
    fmt = "{:<20} {:>12} {:>12}"
    print(fmt.format("항목", "버전 A (기존)", "버전 B (개선)"))
    print("-" * 46)
    print(fmt.format("학습 에피소드 수",
                     f"{stats_a['episodes']:,}",
                     f"{stats_b['episodes']:,}"))
    print(fmt.format("총 경험 수집량",
                     f"{stats_a['total_steps']:,}",
                     f"{stats_b['total_steps']:,}"))
    print(fmt.format("CE 폐기 경험 수",
                     f"{stats_a['ce_blocked']:,}",
                     f"{stats_b['ce_blocked']:,}"))
    print(fmt.format("CE 폐기율",
                     "0.00%",
                     f"{ce_discard_rate:.2%}"))
    print(fmt.format("유효 경험 수",
                     f"{stats_a['valid_exp']:,}",
                     f"{stats_b['valid_exp']:,}"))
    print(fmt.format("도착 에피소드 수",
                     f"{stats_a['arrival_count']:,}",
                     f"{stats_b['arrival_count']:,}"))
    print("-" * 46)
    print(fmt.format("최종 도착률 (100ep)",
                     f"{final_arr_a:.4f}",
                     f"{final_arr_b:.4f}"))
    print(fmt.format("최종 충돌률 (100ep)",
                     f"{final_col_a:.4f}",
                     f"{final_col_b:.4f}"))
    print(fmt.format(f"유효경험 기준 도착률",
                     f"{final_arr_a:.4f}",
                     f"{rate_b_at_match:.4f}"))
    print(f"  ※ 유효경험 기준: 기존DDM 유효경험 {target_valid:,}개 도달 시점"
          f" (개선DDM Ep {ep_b_at_match})")
    print(sep)

    # ── JSON ─────────────────────────────────────────────────────────────────
    results_json = {
        "version_a": {
            "description": "기존 DDM – 모든 경험 저장 + PGD ε=0.03 공격",
            "episodes":               stats_a["episodes"],
            "total_steps":            stats_a["total_steps"],
            "ce_blocked":             stats_a["ce_blocked"],
            "valid_exp":              stats_a["valid_exp"],
            "arrival_count":          stats_a["arrival_count"],
            "final_arrival_rate":     round(final_arr_a, 4),
            "final_collision_rate":   round(final_col_a, 4),
            "arrival_history":        [round(v, 4) for v in arr_a],
            "collision_history":      [round(v, 4) for v in col_a],
        },
        "version_b": {
            "description": "개선 DDM – CE 필터링 + RISK 페널티 + PGD ε=0.03 공격",
            "episodes":               stats_b["episodes"],
            "total_steps":            stats_b["total_steps"],
            "ce_blocked":             stats_b["ce_blocked"],
            "ce_discard_rate":        round(ce_discard_rate, 4),
            "valid_exp":              stats_b["valid_exp"],
            "arrival_count":          stats_b["arrival_count"],
            "final_arrival_rate":     round(final_arr_b, 4),
            "final_collision_rate":   round(final_col_b, 4),
            "valid_exp_match_ep":     ep_b_at_match,
            "arrival_rate_at_match":  round(rate_b_at_match, 4),
            "arrival_history":        [round(v, 4) for v in arr_b],
            "collision_history":      [round(v, 4) for v in col_b],
        },
        "config": {
            "episodes_a":             EPISODES_A,
            "episodes_b":             EPISODES_B,
            "max_steps_per_episode":  MAX_STEPS,
            "space_km":               [SPACE_W, SPACE_H],
            "step_size_km":           STEP_SIZE,
            "goal_radius_km":         GOAL_RADIUS,
            "obstacle_radius_km":     OBS_RADIUS,
            "pgd_epsilon":            0.03,
            "pgd_d_error_rate":       PGD_D_ERR_RATE,
            "pgd_theta_error_deg":    PGD_T_ERR_DEG,
            "ce_shift_threshold_px":  CE_SHIFT_THRESH,
            "risk_dist_km":           RISK_DIST,
            "replay_capacity":        MEMORY_CAPACITY,
            "batch_size":             BATCH_SIZE,
            "gamma":                  GAMMA,
            "lr":                     LR,
            "target_update_freq":     TARGET_UPDATE_FREQ,
        },
    }
    out_json = RESULTS_DIR / "dqn_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    print(f"\n수치 저장 → {out_json}")

    # ── 그래프 (2×1 서브플롯) ────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=False)

    # ── (상단) 에피소드 기준 도착률 비교 ──────────────────────────────────────
    eps_a = list(range(1, EPISODES_A + 1))
    eps_b = list(range(1, EPISODES_B + 1))

    ax1.plot(eps_a, arr_a, color="red",  linewidth=1.5, alpha=0.9,
             label=f"버전 A (기존 DDM, {EPISODES_A}ep) – 최종 {final_arr_a:.3f}")
    ax1.plot(eps_b, arr_b, color="blue", linewidth=1.5, alpha=0.9,
             label=f"버전 B (개선 DDM, {EPISODES_B}ep) – 최종 {final_arr_b:.3f}")

    ax1.set_ylabel("최근 100ep 도착률", fontsize=11)
    ax1.set_title(
        "기존 DDM vs 개선 DDM 학습 성능 비교 – 에피소드 기준\n"
        f"(PGD ε=0.03 공격 환경 | 개선 DDM CE 임계값={CE_SHIFT_THRESH}px)",
        fontsize=12, fontweight="bold",
    )
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, EPISODES_B)
    ax1.set_ylim(0.0, 1.0)
    ax1.axvline(EPISODES_A, color="red", linestyle="--",
                linewidth=0.9, alpha=0.5, label="_버전A 종료")
    ax1.text(EPISODES_A + 20, 0.02, f"버전A\n종료", color="red",
             fontsize=7.5, alpha=0.7, va="bottom")

    # ── (하단) 유효 경험 기준 도착률 비교 ─────────────────────────────────────
    ax2.plot(stats_a["valid_exp_hist"], arr_a, color="red",  linewidth=1.5, alpha=0.9,
             label=f"버전 A (기존 DDM) – 최종 {final_arr_a:.3f}")
    ax2.plot(stats_b["valid_exp_hist"], arr_b, color="blue", linewidth=1.5, alpha=0.9,
             label=f"버전 B (개선 DDM) – 유효경험={stats_b['valid_exp']:,}")

    # 기준선: 기존 DDM의 유효 경험 수
    ax2.axvline(target_valid, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)
    ax2.scatter([target_valid], [rate_b_at_match], color="blue",
                s=70, zorder=5, marker="D",
                label=f"개선 DDM @ 동일 유효경험 → {rate_b_at_match:.3f}")
    ax2.scatter([target_valid], [final_arr_a], color="red",
                s=70, zorder=5, marker="D",
                label=f"기존 DDM @ 동일 유효경험 → {final_arr_a:.3f}")
    ax2.text(target_valid + stats_a["valid_exp"] * 0.01, 0.02,
             f"기준\n{target_valid:,}개", color="gray",
             fontsize=7.5, alpha=0.8, va="bottom")

    ax2.set_xlabel("누적 유효 경험 수 (경험)", fontsize=11)
    ax2.set_ylabel("최근 100ep 도착률", fontsize=11)
    ax2.set_title(
        "기존 DDM vs 개선 DDM 학습 성능 비교 – 유효 경험 기준 (공정 비교)",
        fontsize=12, fontweight="bold",
    )
    ax2.legend(fontsize=9.5, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.0, 1.0)

    plt.tight_layout(rect=[0, 0, 1, 1])
    out_png = RESULTS_DIR / "dqn_comparison_v2.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"그래프 저장 → {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("DQN 비교: 기존 DDM vs 개선 DDM (v2)")
    print(f"환경: {SPACE_W}km × {SPACE_H}km | 장애물 {len(OBSTACLES)}개 (r={OBS_RADIUS}km)")
    print(f"버전 A: {EPISODES_A}에피소드 | 버전 B: {EPISODES_B}에피소드")
    print(f"최대 스텝: {MAX_STEPS} | PGD ε=0.03: D_obs ±{PGD_D_ERR_RATE*100:.2f}%, θ ±{PGD_T_ERR_DEG}°")
    print("=" * 60)

    arr_a, col_a, stats_a = train_version_a()
    arr_b, col_b, stats_b = train_version_b()
    save_results(arr_a, col_a, arr_b, col_b, stats_a, stats_b)

    threshold_analysis()

    print("\n완료.")


if __name__ == "__main__":
    main()
