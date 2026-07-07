"""비행 경로 애니메이션: V9 (확률적 PGD 공격 50%).

V9 핵심 특징:
  - 매 스텝 50% 확률로 PGD 공격 적용 (is_attacked = random.random() < 0.5)
  - 공격 적용:   c_state, shift = pgd_attack(state, ...)
  - 공격 미적용: c_state = state, shift = 0.0

시각화:
  - 경로: 공격 없는 스텝 → 옅은 회색 점, 공격 있는 스텝 → 청색 점
  - PGD 오염 점 (주황): is_attacked=True인 스텝에서만 shift 크기에 비례
  - 범례: "v9: 매 스텝 50% 확률로만 오염 발생" 추가

정책: results/v9_policy.pth 로드 — 학습 없음, epsilon=0.05 고정
에피소드 선택: 목표 도달 시까지 최대 20회 시도, 없으면 최근접 에피소드 사용
저장: results/flight_animation_v9.gif  (8fps)

제약:
  - animate_flight_v5.py 원본 수정 없음
  - 기존 results 파일 덮어쓰기 없음
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as manim
import matplotlib.font_manager as fm
import numpy as np
import torch

# ── 한글 폰트 ───────────────────────────────────────────────────────────────
_prefer = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'Gulim', 'Dotum']
_avail  = {f.name for f in fm.fontManager.ttflist}
for _f in _prefer:
    if _f in _avail:
        plt.rcParams['font.family'] = _f
        break
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

V9_WEIGHTS = RESULTS_DIR / "v9_policy.pth"

from dqn_comparison import DQNAgent, pgd_attack, MAX_STEPS, OBS_RADIUS, GOAL_RADIUS
from env_test import DroneEnvN, OBSTACLES_8, SPACE_W, SPACE_H

# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════

ATTACK_PROB    = 0.5    # v9.py와 동일
EPSILON_EVAL   = 0.05   # v9.py evaluate_policy와 동일

N_SCENARIOS    = 5
SCENARIO_SEED  = 2025
ATTACK_SEED    = 8888
MAX_ATTEMPTS   = 20     # 목표 도달 에피소드 탐색 최대 시도
ANIM_FPS       = 8
PAUSE_FRAMES   = 25

# 시각화
DOT_MIN_SHIFT      = 2.0
DOT_SIZE_SCALE     = 5.0
DOT_ALPHA          = 0.70
_COLOR_DRONE       = "steelblue"
_COLOR_ARRIVED     = "limegreen"
_COLOR_FAIL        = "crimson"
_COLOR_PATH_ATTACK = "steelblue"     # 공격 스텝 경로 점
_COLOR_PATH_CLEAN  = "#b0b8c4"       # 공격 없는 스텝 경로 점 (옅은 회색)
_BORDER_MARGIN     = 2.0

_RESULT_KR = {
    "arrived":       "도착 성공",
    "collision":     "충돌",
    "out_of_bounds": "이탈",
    "timeout":       "시간초과",
}
_RESULT_COLOR = {
    "arrived":       "darkgreen",
    "collision":     "crimson",
    "out_of_bounds": "saddlebrown",
    "timeout":       "gray",
}

Scenario = dict
# FrameData 필드: pos, heading, shift, is_attacked
FrameData = dict


# ══════════════════════════════════════════════════════════════════════════════
# 1. 정책 로드 (학습 없음)
# ══════════════════════════════════════════════════════════════════════════════

def load_v9_agent() -> DQNAgent:
    """v9_policy.pth를 로드해 추론 전용 에이전트를 반환한다.

    파일이 없으면 오류 메시지 출력 후 종료한다.
    """
    if not V9_WEIGHTS.exists():
        print(f"\n[오류] v9_policy.pth를 찾을 수 없습니다: {V9_WEIGHTS}")
        print("  → dqn_comparison_v9.py를 먼저 실행해 가중치를 생성하세요.")
        sys.exit(1)

    agent = DQNAgent()
    agent.policy_net.load_state_dict(
        torch.load(V9_WEIGHTS, map_location="cpu")
    )
    agent.policy_net.eval()   # BatchNorm / Dropout 등 추론 모드 설정
    agent.epsilon = EPSILON_EVAL
    print(f"  v9_policy.pth 로드 완료 | epsilon={EPSILON_EVAL} | gradient 업데이트 없음")
    return agent


# ══════════════════════════════════════════════════════════════════════════════
# 2. 시나리오 생성
# ══════════════════════════════════════════════════════════════════════════════

def get_scenarios(n: int = N_SCENARIOS) -> list[Scenario]:
    """animate_flight_v5.py와 동일한 SCENARIO_SEED로 시나리오를 생성한다."""
    random.seed(SCENARIO_SEED)
    np.random.seed(SCENARIO_SEED)

    env  = DroneEnvN(list(OBSTACLES_8), fallback_pos=[1.0, 1.0])
    scs: list[Scenario] = []
    tries = 0
    while len(scs) < n and tries < 10_000:
        tries += 1
        env.reset()
        px, py = env.pos
        gx, gy = env.goal
        if (px < _BORDER_MARGIN or px > SPACE_W - _BORDER_MARGIN or
                py < _BORDER_MARGIN or py > SPACE_H - _BORDER_MARGIN or
                gx < _BORDER_MARGIN or gx > SPACE_W - _BORDER_MARGIN or
                gy < _BORDER_MARGIN or gy > SPACE_H - _BORDER_MARGIN):
            continue
        scs.append({
            "pos":     list(env.pos),
            "heading": env.heading,
            "goal":    list(env.goal),
        })
    return scs


# ══════════════════════════════════════════════════════════════════════════════
# 3. 에피소드 기록 (v9 확률적 공격 + is_attacked 필드 추가)
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_episode(
    agent: DQNAgent,
    scenario: Scenario,
    seed: int,
) -> tuple[list[FrameData], str]:
    """v9 방식으로 단일 에피소드를 실행하고 프레임 데이터와 결과를 반환한다."""
    random.seed(seed)
    np.random.seed(seed)

    env         = DroneEnvN(list(OBSTACLES_8), fallback_pos=[1.0, 1.0])
    env.pos     = list(scenario["pos"])
    env.heading = scenario["heading"]
    env.goal    = list(scenario["goal"])
    env.steps   = 0
    state       = env._state()

    # 첫 프레임: 시작 위치, 공격 없음
    frames: list[FrameData] = [{
        "pos":         list(env.pos),
        "heading":     env.heading,
        "shift":       0.0,
        "is_attacked": False,
    }]
    result = "timeout"

    for _ in range(MAX_STEPS):
        D_obs_km, theta_obs_deg = env.get_raw_obstacle()

        # v9 핵심: 매 스텝 50% 확률로 공격 결정
        is_attacked = random.random() < ATTACK_PROB
        if is_attacked:
            c_state, shift = pgd_attack(state, D_obs_km, theta_obs_deg)
        else:
            c_state, shift = state, 0.0   # 원본 상태 그대로

        action = agent.act(c_state)
        next_state, _, _, info = env.step(action)

        frames.append({
            "pos":         list(env.pos),
            "heading":     env.heading,
            "shift":       float(shift),
            "is_attacked": is_attacked,
        })
        state = next_state

        if info["goal_reached"]:
            result = "arrived";       break
        if info["collision"]:
            result = "collision";     break
        if info["out_of_bounds"]:
            result = "out_of_bounds"; break

    return frames, result


def record_episode(
    agent: DQNAgent,
    scenario: Scenario,
    base_seed: int,
) -> tuple[list[FrameData], str, int]:
    """최대 MAX_ATTEMPTS번 시도해 목표 도달 에피소드를 찾아 반환한다.

    Returns:
        (frames, result, attempts_used)
        도달 성공 에피소드가 없으면 목표에 가장 근접했던 에피소드 반환.
    """
    best_frames:    list[FrameData] = []
    best_result:    str             = "timeout"
    best_final_dist: float          = float("inf")
    gx, gy = scenario["goal"]

    for attempt in range(MAX_ATTEMPTS):
        seed = base_seed + attempt * 31
        frames, result = _run_one_episode(agent, scenario, seed)

        if result == "arrived":
            return frames, result, attempt + 1

        fx, fy   = frames[-1]["pos"]
        dist     = math.dist([fx, fy], [gx, gy])
        if dist < best_final_dist:
            best_final_dist = dist
            best_frames     = frames
            best_result     = result

    return best_frames, best_result, MAX_ATTEMPTS


# ══════════════════════════════════════════════════════════════════════════════
# 4. 정적 배경 그리기 (animate_flight_v5.py와 동일)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_static(ax: plt.Axes, scenario: Scenario) -> None:
    ax.set_xlim(0, SPACE_W)
    ax.set_ylim(0, SPACE_H)
    ax.set_aspect("equal")
    ax.set_facecolor("#f0f4f8")
    ax.grid(True, alpha=0.18, linewidth=0.5)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([0, 6, 12, 18])
    ax.tick_params(labelsize=7)

    for ox, oy in OBSTACLES_8:
        ax.add_patch(mpatches.Circle(
            (ox, oy), OBS_RADIUS,
            facecolor="#ff4444", alpha=0.18,
            edgecolor="#cc0000", linewidth=0.8, zorder=2,
        ))
        ax.plot(ox, oy, "+", color="#cc0000", markersize=6,
                markeredgewidth=0.9, zorder=3)

    gx, gy = scenario["goal"]
    ax.add_patch(mpatches.Circle(
        (gx, gy), GOAL_RADIUS,
        facecolor="gold", alpha=0.20,
        edgecolor="goldenrod", linewidth=0.9, zorder=2,
    ))
    ax.plot(gx, gy, "*", color="goldenrod", markersize=15,
            markeredgecolor="#996600", markeredgewidth=0.5, zorder=7)

    sx, sy = scenario["pos"]
    ax.plot(sx, sy, "^", color="lightsteelblue", markersize=8,
            markeredgecolor="navy", markeredgewidth=0.6, alpha=0.55, zorder=3)


def _dot_size(shift: float) -> float:
    return float(np.clip(shift * DOT_SIZE_SCALE, 8, 220))


# ══════════════════════════════════════════════════════════════════════════════
# 5. 애니메이션 클래스
# ══════════════════════════════════════════════════════════════════════════════

class DroneAnimV9:
    """V9 단일 패널 애니메이션 상태 관리.

    경로 색상 구분:
      공격 없는 스텝 (is_attacked=False) → 옅은 회색 scatter 점
      공격 있는 스텝 (is_attacked=True)  → 청색 scatter 점

    오염 점 (주황, shift 비례 크기):
      is_attacked=True이고 shift >= DOT_MIN_SHIFT인 스텝에서만 표시.
    """

    def __init__(
        self,
        ax: plt.Axes,
        scenario: Scenario,
        frames: list[FrameData],
        result: str,
        label: str,
    ) -> None:
        self.ax       = ax
        self.frames   = frames
        self.result   = result
        self.scenario = scenario

        _draw_static(ax, scenario)

        ax.text(0.03, 0.97, label,
                transform=ax.transAxes, fontsize=8.5,
                fontweight="bold", color="steelblue",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        sx, sy = scenario["pos"]

        # 경로 점 — 공격 없는 구간 (옅은 회색)
        self.path_clean_sc = ax.scatter(
            [], [], s=14, color=_COLOR_PATH_CLEAN,
            alpha=0.65, linewidths=0, zorder=4,
        )
        # 경로 점 — 공격 구간 (청색)
        self.path_attack_sc = ax.scatter(
            [], [], s=14, color=_COLOR_PATH_ATTACK,
            alpha=0.70, linewidths=0, zorder=5,
        )

        # 드론 마커
        self.drone_sc = ax.scatter(
            [sx], [sy], s=130, marker="^",
            color=_COLOR_DRONE, edgecolors="navy", linewidths=0.8, zorder=8,
        )
        self.arrow = ax.quiver(
            sx, sy,
            math.cos(math.radians(scenario["heading"])) * 0.55,
            math.sin(math.radians(scenario["heading"])) * 0.55,
            color=_COLOR_DRONE, scale=1, scale_units="xy",
            width=0.014, headwidth=3.5, headlength=3.5, zorder=9, alpha=0.90,
        )

        # PGD 오염 점 (is_attacked=True 스텝에서만)
        self.dot_sc = ax.scatter(
            [], [], s=[], c="darkorange", alpha=DOT_ALPHA, linewidths=0, zorder=6,
        )

        self.status_txt = ax.text(
            0.97, 0.03, "", transform=ax.transAxes, fontsize=7.5,
            ha="right", va="bottom", color="gray", fontweight="bold",
        )
        self.step_txt = ax.text(
            0.03, 0.03, "Step 0", transform=ax.transAxes, fontsize=7,
            ha="left", va="bottom", color="dimgray",
        )
        # 공격 비율 실시간 표시
        self.atk_txt = ax.text(
            0.50, 0.03, "", transform=ax.transAxes, fontsize=7,
            ha="center", va="bottom", color="#c05a00",
        )

        self._prev_fi = -1   # 중복 누적 방지

    def update(self, frame_idx: int) -> None:
        fi     = min(frame_idx, len(self.frames) - 1)
        fd     = self.frames[fi]
        px, py = fd["pos"]
        hdg    = fd["heading"]
        shift  = fd["shift"]
        is_atk = fd["is_attacked"]
        done   = frame_idx >= len(self.frames)

        # ── 경로 점 누적 (프레임이 새로 진행된 경우에만) ─────────────────
        if fi > 0 and fi != self._prev_fi and not done:
            self._prev_fi = fi
            # 현재까지 frames[1..fi]에서 공격/비공격 위치 분류
            clean_pts  = [(f["pos"][0], f["pos"][1])
                          for f in self.frames[1:fi + 1]
                          if not f["is_attacked"]]
            attack_pts = [(f["pos"][0], f["pos"][1])
                          for f in self.frames[1:fi + 1]
                          if f["is_attacked"]]

            if clean_pts:
                self.path_clean_sc.set_offsets(clean_pts)
            if attack_pts:
                self.path_attack_sc.set_offsets(attack_pts)

            # 공격 비율 표시
            total = len(clean_pts) + len(attack_pts)
            if total > 0:
                atk_pct = len(attack_pts) / total
                self.atk_txt.set_text(
                    f"공격 {len(attack_pts)} / 전체 {total} 스텝"
                    f" ({atk_pct:.0%})"
                )

        # ── 드론 마커 갱신 ────────────────────────────────────────────────
        if done:
            dc = _COLOR_ARRIVED if self.result == "arrived" else _COLOR_FAIL
        else:
            dc = _COLOR_DRONE
        self.drone_sc.set_offsets([[px, py]])
        self.drone_sc.set_color(dc)

        dx = math.cos(math.radians(hdg)) * 0.50
        dy = math.sin(math.radians(hdg)) * 0.50
        self.arrow.set_offsets([[px, py]])
        self.arrow.set_UVC(dx, dy)
        self.arrow.set_color(dc)

        # ── PGD 오염 점: is_attacked=True이고 shift >= DOT_MIN_SHIFT만 ──
        if not done and is_atk and shift >= DOT_MIN_SHIFT:
            dot_pts = [
                (f["pos"][0], f["pos"][1])
                for f in self.frames[1:fi + 1]
                if f["is_attacked"] and f["shift"] >= DOT_MIN_SHIFT
            ]
            dot_sizes = [
                _dot_size(f["shift"])
                for f in self.frames[1:fi + 1]
                if f["is_attacked"] and f["shift"] >= DOT_MIN_SHIFT
            ]
            if dot_pts:
                self.dot_sc.set_offsets(dot_pts)
                self.dot_sc.set_sizes(dot_sizes)

        # ── 하단 텍스트 갱신 ──────────────────────────────────────────────
        self.step_txt.set_text(f"Step {fi}")

        if done:
            rk = _RESULT_KR.get(self.result, self.result)
            rc = _RESULT_COLOR.get(self.result, "gray")
            self.status_txt.set_text(rk)
            self.status_txt.set_color(rc)
        else:
            if is_atk:
                intensity = "강" if shift >= 20 else "중" if shift >= 10 else "약"
                self.status_txt.set_text(f"PGD shift={shift:.1f}px [{intensity}]")
                self.status_txt.set_color("darkorange" if shift >= 10 else "dimgray")
            else:
                self.status_txt.set_text("공격 없음 (원본 상태)")
                self.status_txt.set_color("steelblue")


# ══════════════════════════════════════════════════════════════════════════════
# 6. 5-시나리오 순차 애니메이션 빌드
# ══════════════════════════════════════════════════════════════════════════════

EpData = tuple  # (scenario, frames, result, attempts)


def build_animation_v9(all_ep: list[EpData]) -> manim.FuncAnimation:
    """5개 시나리오를 순서대로 재생하는 FuncAnimation을 반환한다."""

    frame_schedule: list[tuple[int, int]] = []
    for sc_idx, (_, frames, _, _) in enumerate(all_ep):
        ep_len = len(frames) + PAUSE_FRAMES
        for lf in range(ep_len):
            frame_schedule.append((sc_idx, lf))

    total_frames = len(frame_schedule)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 8.5))
    fig.patch.set_facecolor("#e8ecf0")

    suptitle = fig.suptitle(
        f"V9  |  시나리오 1 / {N_SCENARIOS}",
        fontsize=12, fontweight="bold", y=0.98,
    )

    legend_handles = [
        mpatches.Patch(color=_COLOR_DRONE,        alpha=0.85, label="드론 (비행 중)"),
        mpatches.Patch(color=_COLOR_ARRIVED,       alpha=0.85, label="도착 성공"),
        mpatches.Patch(color=_COLOR_FAIL,          alpha=0.85, label="충돌 / 이탈"),
        mpatches.Patch(color="#ff4444",            alpha=0.30, label="장애물 (r=1.5km)"),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="goldenrod", markersize=11, label="목적지"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_COLOR_PATH_ATTACK, markersize=6,
                   label="경로 (공격 스텝)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_COLOR_PATH_CLEAN, markersize=6,
                   label="경로 (공격 없는 스텝)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="darkorange", markersize=5,
                   label="PGD 오염 (소) shift ~5px"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="darkorange", markersize=9,
                   label="PGD 오염 (중) shift ~15px"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="darkorange", markersize=13,
                   label="PGD 오염 (대) shift ~30px"),
        mpatches.Patch(color="white", alpha=0,
                       label="v9: 매 스텝 50% 확률로만 오염 발생"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=3, fontsize=7.0,
        bbox_to_anchor=(0.5, 0.0),
        framealpha=0.92,
    )

    frame_txt = fig.text(
        0.5, 0.945, "", ha="center", va="top", fontsize=7.5, color="dimgray",
    )

    plt.tight_layout(rect=[0, 0.16, 1, 0.96])

    state = {"sc_idx": -1, "anim": None}

    def update(global_frame: int):
        sc_idx, local_f = frame_schedule[global_frame]

        if sc_idx != state["sc_idx"]:
            ax.clear()
            sc, frames, result, attempts = all_ep[sc_idx]
            atk_cnt   = sum(1 for f in frames[1:] if f["is_attacked"])
            total_stp = len(frames) - 1
            label = (
                f"V9  시나리오 {sc_idx + 1}  |  "
                f"{len(frames) - 1}스텝 / {result}\n"
                f"공격 적용: {atk_cnt}/{total_stp}스텝 "
                f"({atk_cnt/max(total_stp,1):.0%})  "
                f"시도 {attempts}회"
            )
            state["anim"]   = DroneAnimV9(ax, sc, frames, result, label)
            state["sc_idx"] = sc_idx
            suptitle.set_text(
                f"V9 확률적 PGD 공격 (50%)  |  시나리오 {sc_idx + 1} / {N_SCENARIOS}"
            )

        state["anim"].update(local_f)
        frame_txt.set_text(f"Global frame {global_frame + 1} / {total_frames}")
        return []

    anim = manim.FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000 // ANIM_FPS, blit=False,
    )
    return anim


# ══════════════════════════════════════════════════════════════════════════════
# 7. 메인
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 68)
    print("비행 경로 애니메이션 v9: 확률적 PGD 공격 (50%/스텝)")
    print(f"가중치: {V9_WEIGHTS}")
    print(f"epsilon={EPSILON_EVAL} 고정  |  학습 없음  |  {N_SCENARIOS}개 시나리오")
    print(f"에피소드 탐색: 목표 도달까지 최대 {MAX_ATTEMPTS}회 시도")
    print("=" * 68)

    # ── 가중치 존재 확인 ──────────────────────────────────────────────────
    print(f"\n[0단계] v9_policy.pth 존재 확인...")
    if not V9_WEIGHTS.exists():
        print(f"  [오류] 파일 없음: {V9_WEIGHTS}")
        print("  → dqn_comparison_v9.py를 먼저 실행하세요.")
        return
    print(f"  확인: {V9_WEIGHTS} ({V9_WEIGHTS.stat().st_size // 1024}KB)")

    # ── 에이전트 로드 ──────────────────────────────────────────────────────
    print("\n[1단계] 정책 로드...")
    agent = load_v9_agent()

    # ── 시나리오 생성 ──────────────────────────────────────────────────────
    print("\n[2단계] 시나리오 생성...")
    scenarios = get_scenarios(N_SCENARIOS)
    for i, sc in enumerate(scenarios):
        print(f"  #{i+1}: 시작={[round(v,1) for v in sc['pos']]}"
              f" → 목적지={[round(v,1) for v in sc['goal']]}")

    # ── 시험 에피소드: is_attacked 비율과 목표 도달 확인 ─────────────────
    print("\n[3단계] 시험 에피소드 (시나리오 #1, seed=9999)...")
    test_frames, test_result = _run_one_episode(agent, scenarios[0], seed=9999)
    test_steps   = len(test_frames) - 1
    test_atk_cnt = sum(1 for f in test_frames[1:] if f["is_attacked"])
    test_atk_rate = test_atk_cnt / max(test_steps, 1)
    print(f"  결과: {test_result}  |  총 스텝: {test_steps}")
    print(f"  공격 적용 스텝: {test_atk_cnt}  |  비율: {test_atk_rate:.1%}"
          f"  (기대값 ~50%)")
    print(f"  {'[정상] 50%에 근접' if 0.35 <= test_atk_rate <= 0.65 else '[경고] 50%에서 크게 벗어남'}")

    # ── 에피소드 기록 ──────────────────────────────────────────────────────
    print("\n[4단계] 5개 시나리오 에피소드 기록 (목표 도달 우선)...")
    all_ep: list[EpData] = []
    results_summary: list[str] = []

    for i, sc in enumerate(scenarios):
        base_seed = ATTACK_SEED + i * 17
        frames, result, attempts = record_episode(agent, sc, base_seed)

        atk_cnt   = sum(1 for f in frames[1:] if f["is_attacked"])
        total_stp = len(frames) - 1
        dot_cnt   = sum(1 for f in frames[1:]
                        if f["is_attacked"] and f["shift"] >= DOT_MIN_SHIFT)
        print(f"  시나리오 #{i+1}: {total_stp}스텝 / {result}"
              f" | 공격 {atk_cnt}/{total_stp}({atk_cnt/max(total_stp,1):.0%})"
              f" | 오염점 {dot_cnt}개"
              f" | 시도 {attempts}회")
        all_ep.append((sc, frames, result, attempts))
        results_summary.append(result)

    # ── 결과 요약 ──────────────────────────────────────────────────────────
    _KR = _RESULT_KR
    print("\n" + "─" * 60)
    print(f"{'시나리오':<10} {'결과':<12} {'스텝수':<8} {'공격비율':<10} {'오염점'}")
    print("─" * 60)
    for i, (sc, frames, result, _) in enumerate(all_ep):
        total_stp = len(frames) - 1
        atk_cnt   = sum(1 for f in frames[1:] if f["is_attacked"])
        dot_cnt   = sum(1 for f in frames[1:]
                        if f["is_attacked"] and f["shift"] >= DOT_MIN_SHIFT)
        print(f"  #{i+1}         {_KR[result]:<10}  {total_stp:<6}  "
              f"{atk_cnt/max(total_stp,1):.0%}({atk_cnt}/{total_stp})  "
              f"{dot_cnt}개")
    print("─" * 60)
    arrived = sum(1 for r in results_summary if r == "arrived")
    print(f"  도착 성공: {arrived} / {N_SCENARIOS}")
    print("─" * 60)

    # ── 애니메이션 생성·저장 ───────────────────────────────────────────────
    print("\n[5단계] 애니메이션 생성 중...")
    anim = build_animation_v9(all_ep)

    out_gif = RESULTS_DIR / "flight_animation_v9.gif"
    writer  = manim.PillowWriter(fps=ANIM_FPS, metadata={"loop": 0})
    print(f"GIF 저장 중 → {out_gif}  (fps={ANIM_FPS})")
    anim.save(str(out_gif), writer=writer, dpi=90)
    plt.close("all")

    try:
        from PIL import Image
        with Image.open(out_gif) as img:
            w, h = img.size
            n_fr = getattr(img, "n_frames", "?")
        kb = out_gif.stat().st_size // 1024
        print(f"완료: {w}x{h}px  {n_fr}프레임  {kb}KB → {out_gif}")
    except ImportError:
        kb = out_gif.stat().st_size // 1024
        print(f"완료: {kb}KB → {out_gif}")


if __name__ == "__main__":
    main()
