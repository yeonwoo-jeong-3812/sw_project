"""비행 경로 애니메이션: V5-A (이진 CE) vs V5-B (연속 CE 가중치).

V5 변경점 (V4 대비):
  - 시간 페널티: -0.01 / 비종료 스텝
  - 거리 보상: +0.01 (목적지에 가까워짐) / -0.01 (멀어짐)
  - 종료 보상은 동일: 도착 +1.0 / 충돌·이탈 -1.0

화면 구성:
  왼쪽: V5-A 드론 – 이진 CE 차단 (shift > 10px → 저장 안 함)
  오른쪽: V5-B 드론 – 연속 가중치 (max(0.05, 1 - shift/50))

시각화:
  - 드론: 파란 삼각형 (도착=초록, 충돌=빨강)
  - PGD 오염 점: 경로 위 주황색 원, 크기 ∝ shift 크기
  - 5개 시나리오를 순서대로 재생하는 단일 GIF

환경: 8장애물, 랜덤 시작/목적지 (V5 실험 동일)
저장: results/flight_animation_v5.gif  (8fps)
"""

from __future__ import annotations

import math
import random
import sys
from collections import deque
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

# ── 공유 컴포넌트 import ────────────────────────────────────────────────────
from dqn_comparison import DQNAgent, pgd_attack, MAX_STEPS, OBS_RADIUS, GOAL_RADIUS
from env_test import DroneEnvN, OBSTACLES_8, SPACE_W, SPACE_H

# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════

CE_BINARY_THRESH = 10.0
CE_SHIFT_MAX     = 50.0
EPISODES_TRAIN   = 1000
N_SCENARIOS      = 5
SCENARIO_SEED    = 2025
ATTACK_SEED      = 8888
ANIM_FPS         = 8
PAUSE_FRAMES     = 25

# V5 보상 파라미터
TIME_PENALTY  = -0.01
DIST_APPROACH = +0.01
DIST_RECEDE   = -0.01

# 오염 점 시각화
DOT_MIN_SHIFT  = 2.0
DOT_SIZE_SCALE = 5.0
DOT_ALPHA      = 0.70


# ══════════════════════════════════════════════════════════════════════════════
# 1. 에이전트 학습 (V5 보상 포함)
# ══════════════════════════════════════════════════════════════════════════════

def _train_agent_v5(use_binary_ce: bool, tag: str) -> DQNAgent:
    """V5 보상 + CE 전략으로 DQNAgent를 학습하고 반환한다.

    use_binary_ce=True  → shift > 10px 차단 (V5-A)
    use_binary_ce=False → max(0.05, 1-shift/50) 확률 deposit (V5-B)
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    env   = DroneEnvN(OBSTACLES_8, fallback_pos=[1.0, 1.0])
    agent = DQNAgent()
    wa    = deque(maxlen=100)

    for ep in range(EPISODES_TRAIN):
        state     = env.reset()
        arrived   = False
        prev_dist = math.dist(env.pos, env.goal)

        for _ in range(MAX_STEPS):
            D_obs, theta_obs = env.get_raw_obstacle()
            c_state, shift   = pgd_attack(state, D_obs, theta_obs)

            action = agent.act(c_state)
            next_state, reward, done, info = env.step(action)

            # ── V5 보상 추가 (비종료 스텝만) ──────────────────────────────
            terminal = info["goal_reached"] or info["collision"] or info["out_of_bounds"]
            if not terminal:
                reward += TIME_PENALTY
                curr_dist = math.dist(env.pos, env.goal)
                if curr_dist < prev_dist:
                    reward += DIST_APPROACH
                elif curr_dist > prev_dist:
                    reward += DIST_RECEDE
                prev_dist = curr_dist

            # ── CE deposit 판정 ────────────────────────────────────────────
            if use_binary_ce:
                should_deposit = (shift <= CE_BINARY_THRESH)
            else:
                ratio = max(0.05, 1.0 - shift / CE_SHIFT_MAX)
                should_deposit = (random.random() < ratio)

            if should_deposit:
                D_n, t_n  = env.get_raw_obstacle()
                c_next, _ = pgd_attack(next_state, D_n, t_n)
                agent.push(c_state, action, reward, c_next, done)

            agent.update()
            state = next_state

            if info["goal_reached"]:
                arrived = True
                break
            if done:
                break

        agent.decay_epsilon()
        wa.append(1 if arrived else 0)

        if (ep + 1) % 200 == 0:
            print(f"  {tag} Ep {ep+1:4d} | 도착률={sum(wa)/len(wa):.3f}"
                  f" | ε={agent.epsilon:.3f}")

    agent.epsilon = 0.0
    return agent


def train_v5a_agent() -> DQNAgent:
    print("\n[1/2] V5-A 학습 중 (이진 CE 10px + V5 보상) ...")
    ag = _train_agent_v5(use_binary_ce=True, tag="[V5-A]")
    print("      완료")
    return ag


def train_v5b_agent() -> DQNAgent:
    print("\n[2/2] V5-B 학습 중 (연속 CE 가중치 + V5 보상) ...")
    ag = _train_agent_v5(use_binary_ce=False, tag="[V5-B]")
    print("      완료")
    return ag


# ══════════════════════════════════════════════════════════════════════════════
# 2. 시나리오 생성 및 에피소드 기록
# ══════════════════════════════════════════════════════════════════════════════

Scenario = dict

_BORDER_MARGIN = 2.0


def get_scenarios(n: int = N_SCENARIOS) -> list[Scenario]:
    """경계 여유 _BORDER_MARGIN km를 두고 n개 랜덤 시나리오를 생성한다."""
    random.seed(SCENARIO_SEED)
    np.random.seed(SCENARIO_SEED)

    env  = DroneEnvN(OBSTACLES_8, fallback_pos=[1.0, 1.0])
    scs: list[Scenario] = []
    max_tries = 10_000
    tries     = 0
    while len(scs) < n and tries < max_tries:
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


FrameData = dict


def record_episode(
    agent: DQNAgent,
    scenario: Scenario,
    attack_seed: int,
) -> tuple[list[FrameData], str]:
    """에이전트를 주어진 시나리오에서 실행하고 프레임 데이터를 반환한다."""
    random.seed(attack_seed)
    np.random.seed(attack_seed)

    env         = DroneEnvN(OBSTACLES_8, fallback_pos=[1.0, 1.0])
    env.pos     = list(scenario["pos"])
    env.heading = scenario["heading"]
    env.goal    = list(scenario["goal"])
    env.steps   = 0
    state       = env._state()

    frames: list[FrameData] = [{"pos": list(env.pos), "heading": env.heading, "shift": 0.0}]
    result = "timeout"

    for _ in range(MAX_STEPS):
        D_obs, theta_obs = env.get_raw_obstacle()
        c_state, shift   = pgd_attack(state, D_obs, theta_obs)
        action           = agent.act(c_state)
        next_state, _, _, info = env.step(action)

        frames.append({"pos": list(env.pos), "heading": env.heading, "shift": float(shift)})
        state = next_state

        if info["goal_reached"]:
            result = "arrived";       break
        if info["collision"]:
            result = "collision";     break
        if info["out_of_bounds"]:
            result = "out_of_bounds"; break

    return frames, result


# ══════════════════════════════════════════════════════════════════════════════
# 3. 애니메이션 클래스
# ══════════════════════════════════════════════════════════════════════════════

_COLOR_DRONE   = "steelblue"
_COLOR_ARRIVED = "limegreen"
_COLOR_FAIL    = "crimson"

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


def _dot_size(shift: float) -> float:
    return float(np.clip(shift * DOT_SIZE_SCALE, 8, 220))


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


class DroneAnimV5:
    """V5 단일 패널 애니메이션 상태 관리."""

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

        self.path_line, = ax.plot([], [], color="steelblue",
                                   linewidth=1.0, alpha=0.65, zorder=4)
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
        self.dot_sc = ax.scatter(
            [], [], s=[], c="darkorange", alpha=DOT_ALPHA, linewidths=0, zorder=6,
        )
        self._dot_xs:    list[float] = []
        self._dot_ys:    list[float] = []
        self._dot_sizes: list[float] = []

        self.status_txt = ax.text(
            0.97, 0.03, "", transform=ax.transAxes, fontsize=7.5,
            ha="right", va="bottom", color="gray", fontweight="bold",
        )
        self.step_txt = ax.text(
            0.03, 0.03, "Step 0", transform=ax.transAxes, fontsize=7,
            ha="left", va="bottom", color="dimgray",
        )

    def update(self, frame_idx: int) -> None:
        fi     = min(frame_idx, len(self.frames) - 1)
        fd     = self.frames[fi]
        px, py = fd["pos"]
        hdg    = fd["heading"]
        shift  = fd["shift"]
        done   = frame_idx >= len(self.frames)

        xs = [f["pos"][0] for f in self.frames[:fi + 1]]
        ys = [f["pos"][1] for f in self.frames[:fi + 1]]
        self.path_line.set_data(xs, ys)

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

        if not done and shift >= DOT_MIN_SHIFT:
            self._dot_xs.append(px)
            self._dot_ys.append(py)
            self._dot_sizes.append(_dot_size(shift))
            if self._dot_xs:
                self.dot_sc.set_offsets(list(zip(self._dot_xs, self._dot_ys)))
                self.dot_sc.set_sizes(self._dot_sizes)

        self.step_txt.set_text(f"Step {fi}")
        if done:
            rk = _RESULT_KR.get(self.result, self.result)
            rc = _RESULT_COLOR.get(self.result, "gray")
            self.status_txt.set_text(rk)
            self.status_txt.set_color(rc)
        else:
            intensity = "강" if shift >= 20 else "중" if shift >= 10 else "약"
            self.status_txt.set_text(f"PGD shift={shift:.1f}px [{intensity}]")
            self.status_txt.set_color("darkorange" if shift >= 10 else "gray")


# ══════════════════════════════════════════════════════════════════════════════
# 4. 5-시나리오 순차 애니메이션 빌드
# ══════════════════════════════════════════════════════════════════════════════

EpData = tuple


def build_animation_v5(
    all_ep: list[EpData],
    label_a: str = "V5-A  이진 CE (>10px 차단) + 시간/거리 보상",
    label_b: str = "V5-B  연속 CE 가중치 (max(0.05,1-s/50)) + 시간/거리 보상",
) -> manim.FuncAnimation:
    """5개 시나리오를 순서대로 재생하는 FuncAnimation을 반환한다."""

    frame_schedule: list[tuple[int, int]] = []
    for sc_idx, (_, fa, _, fb, _) in enumerate(all_ep):
        ep_len = max(len(fa), len(fb)) + PAUSE_FRAMES
        for lf in range(ep_len):
            frame_schedule.append((sc_idx, lf))

    total_frames = len(frame_schedule)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(14, 7.5),
        gridspec_kw={"wspace": 0.16},
    )
    fig.patch.set_facecolor("#e8ecf0")

    suptitle = fig.suptitle(
        f"V5-A vs V5-B  |  시나리오 1 / {N_SCENARIOS}",
        fontsize=12, fontweight="bold", y=0.97,
    )

    legend_handles = [
        mpatches.Patch(color=_COLOR_DRONE,   alpha=0.85, label="드론 (비행 중)"),
        mpatches.Patch(color=_COLOR_ARRIVED, alpha=0.85, label="도착 성공"),
        mpatches.Patch(color=_COLOR_FAIL,    alpha=0.85, label="충돌 / 이탈"),
        mpatches.Patch(color="#ff4444",      alpha=0.30, label="장애물 (r=1.5km)"),
        plt.Line2D([0], [0], marker="*", color="w",
                   markerfacecolor="goldenrod", markersize=11, label="목적지"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="darkorange", markersize=5,
                   label="PGD 오염 (소) shift ~5px"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="darkorange", markersize=9,
                   label="PGD 오염 (중) shift ~15px"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="darkorange", markersize=13,
                   label="PGD 오염 (대) shift ~30px"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=4, fontsize=7.5,
        bbox_to_anchor=(0.5, 0.0),
        framealpha=0.92,
    )

    frame_txt = fig.text(
        0.5, 0.92, "", ha="center", va="top", fontsize=7.5, color="dimgray",
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    state = {"sc_idx": -1, "da": None, "db": None}

    def update(global_frame: int):
        sc_idx, local_f = frame_schedule[global_frame]

        if sc_idx != state["sc_idx"]:
            ax_a.clear()
            ax_b.clear()
            sc, fa, ra, fb, rb = all_ep[sc_idx]
            state["da"] = DroneAnimV5(ax_a, sc, fa, ra, label_a)
            state["db"] = DroneAnimV5(ax_b, sc, fb, rb, label_b)
            state["sc_idx"] = sc_idx
            suptitle.set_text(
                f"V5-A vs V5-B  |  시나리오 {sc_idx + 1} / {N_SCENARIOS}"
            )

        state["da"].update(local_f)
        state["db"].update(local_f)
        frame_txt.set_text(f"Global frame {global_frame + 1} / {total_frames}")
        return []

    anim = manim.FuncAnimation(
        fig, update, frames=total_frames,
        interval=1000 // ANIM_FPS, blit=False,
    )
    return anim


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 64)
    print("비행 경로 애니메이션 v5: V5-A (이진 CE) vs V5-B (연속 CE)")
    print(f"환경: 8장애물 / 랜덤 시작·목적지 / {N_SCENARIOS}개 시나리오")
    print(f"V5 보상: 시간패널티 {TIME_PENALTY}/step, 거리보상 +/-{abs(DIST_APPROACH)}/step")
    print(f"FPS: {ANIM_FPS}  |  학습: {EPISODES_TRAIN}ep")
    print("=" * 64)

    # ── 에이전트 학습 ─────────────────────────────────────────────────────
    agent_a = train_v5a_agent()
    agent_b = train_v5b_agent()

    # ── 시나리오 생성 ──────────────────────────────────────────────────────
    scenarios = get_scenarios(N_SCENARIOS)
    print(f"\n시나리오 {N_SCENARIOS}개 생성 완료:")
    for i, sc in enumerate(scenarios):
        print(f"  #{i+1}: 시작={[round(v,1) for v in sc['pos']]}"
              f" → 목적지={[round(v,1) for v in sc['goal']]}")

    # ── 에피소드 기록 + 시나리오별 결과표 ────────────────────────────────
    print("\n에피소드 기록 중...")
    all_ep: list[EpData] = []
    results_a: list[str] = []
    results_b: list[str] = []

    for i, sc in enumerate(scenarios):
        seed = ATTACK_SEED + i * 17
        fa, ra = record_episode(agent_a, sc, seed)
        fb, rb = record_episode(agent_b, sc, seed)
        n_dot_a = sum(1 for f in fa if f["shift"] >= DOT_MIN_SHIFT)
        n_dot_b = sum(1 for f in fb if f["shift"] >= DOT_MIN_SHIFT)
        print(f"  시나리오 #{i+1}: "
              f"V5-A {len(fa)-1}스텝/{ra} (오염점 {n_dot_a}개) | "
              f"V5-B {len(fb)-1}스텝/{rb} (오염점 {n_dot_b}개)")
        all_ep.append((sc, fa, ra, fb, rb))
        results_a.append(ra)
        results_b.append(rb)

    # ── 결과 요약표 출력 ───────────────────────────────────────────────────
    _KR = _RESULT_KR
    print("\n" + "─" * 62)
    print(f"{'시나리오':<8} {'V4 문제':<12} {'V5-A 결과':<14} {'V5-B 결과':<14}")
    print("─" * 62)
    for i in range(N_SCENARIOS):
        sc = scenarios[i]
        s  = [round(v, 1) for v in sc["pos"]]
        g  = [round(v, 1) for v in sc["goal"]]
        v4_issue = "빙빙돌기(시간초과)" if results_a[i] == "timeout" else "-"
        print(f"  #{i+1}  ({s}→{g})")
        print(f"        V4문제={v4_issue:<14}"
              f" V5-A={_KR[results_a[i]]:<10}"
              f" V5-B={_KR[results_b[i]]}")
    print("─" * 62)

    arr_a = sum(1 for r in results_a if r == "arrived")
    arr_b = sum(1 for r in results_b if r == "arrived")
    print(f"  도착 수: V5-A {arr_a}/{N_SCENARIOS}  |  V5-B {arr_b}/{N_SCENARIOS}")
    print("─" * 62)

    # ── 애니메이션 생성·저장 ───────────────────────────────────────────────
    print("\n애니메이션 생성 중...")
    anim = build_animation_v5(all_ep)

    out_gif = RESULTS_DIR / "flight_animation_v5.gif"
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
