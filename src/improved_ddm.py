"""개선된 DDM(Dynamic Decision-Making) 설계 모듈.

FRDDM-DQN 논문의 기존 DDM에 적대적 공격 오염 감지(CE) 범주를 추가한다.

실험 근거 (src/attack.py, src/analyze.py 결과):
  - epsilon=0.01: 평균 bbox 중심 오차 12.9px, D'OtoU 오차율 0.18%, θ 오차 0.17°
  - epsilon=0.03: 평균 bbox 중심 오차 13.8px, D'OtoU 오차율 0.39%, θ 오차 0.38°
  - epsilon=0.05: 평균 bbox 중심 오차 14.1px (최대), D'OtoU 오차율 0.50%
  - epsilon=0.07: 평균 bbox 중심 오차 12.4px, D'OtoU 오차율 0.69% (최대)
  - epsilon=0.10: 평균 bbox 중심 오차 10.7px, D'OtoU 오차율 0.66%

  - vehicle_04에서 epsilon=0.01에서 2→8개로 ghost box 급증 확인
  - epsilon=0.01 이상에서 신뢰도 점수가 0.93→0.82 수준으로 하락
  - 10px 임계값: epsilon=0.01(12.9px)에서 이미 초과 → 조기 감지 가능
  - 20px 임계값: vehicle_04, vehicle_02처럼 큰 오차 케이스 포착용
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════
# 열거형
# ══════════════════════════════════════════════════════════════════════════

class ExperienceType(Enum):
    """DDM 경험 분류 (CE 추가).

    기존 FRDDM-DQN: RE / DE / SE
    개선안:          RE / DE / SE / CE
    """
    RE = auto()   # Resultant Experience  – 목표 도달 / 충돌 결과 포함
    DE = auto()   # Dangerous Experience  – 장애물 근접 위험 상태
    SE = auto()   # Safe Experience       – 안전한 일반 이동
    CE = auto()   # Corrupted Experience  – 적대적 공격 오염 감지


class CorruptionLevel(Enum):
    """오염 심각도.

    STABLE   : 오염 없음 (score≥0.7, shift≤10px)
    RISK     : 경미한 오염 (score<0.7 또는 10<shift≤20px)
               → epsilon=0.01 수준(평균 12.9px)에서 주로 발생
    CRITICAL : 심각한 오염 (score<0.5 또는 shift>20px)
               → vehicle_04의 epsilon=0.01 케이스(ghost box 8개) 해당
    """
    STABLE   = auto()
    RISK     = auto()
    CRITICAL = auto()


# ══════════════════════════════════════════════════════════════════════════
# 데이터 클래스
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """Faster R-CNN 단일 검출 결과."""
    box:    list[float]        # [x1, y1, x2, y2]
    score:  float
    center: tuple[float, float]  # (cx, cy)

    @staticmethod
    def from_dict(d: dict) -> "Detection":
        cx, cy = d["centers"][0]
        return Detection(
            box=d["boxes"][0],
            score=d["scores"][0],
            center=(cx, cy),
        )


@dataclass
class Experience:
    """DQN 리플레이 메모리 단위 경험."""
    state:      list[float]
    action:     int
    reward:     float
    next_state: list[float]
    done:       bool
    exp_type:   ExperienceType = ExperienceType.SE
    corruption: CorruptionLevel = CorruptionLevel.STABLE


# ══════════════════════════════════════════════════════════════════════════
# 1. 오염 감지 함수
# ══════════════════════════════════════════════════════════════════════════

# 오염 판단 임계값
_SHIFT_THRESHOLD_RISK     = 10.0   # px – epsilon=0.01에서 평균 12.9px 초과 확인
_SHIFT_THRESHOLD_CRITICAL = 20.0   # px – vehicle_04 ghost box 급증 케이스 기준
_SCORE_THRESHOLD_RISK     = 0.70   # epsilon=0.01~0.03에서 일부 박스 0.82→아래로 하락
_SCORE_THRESHOLD_CRITICAL = 0.50   # 점수 50% 미만: 검출 신뢰 불가 수준


def center_shift(prev: tuple[float, float], curr: tuple[float, float]) -> float:
    """연속 프레임 간 bbox 중심 유클리드 거리 (픽셀)."""
    return math.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)


def assess_corruption(
    current: Detection,
    previous: Optional[Detection] = None,
) -> CorruptionLevel:
    """오염 심각도를 판정한다.

    조건 A (점수 기반):
      - score < 0.50 → CRITICAL
        # epsilon=0.01에서 vehicle_03 허위 박스 score=0.34 관찰
      - score < 0.70 → RISK
        # epsilon=0.01~0.10 구간에서 ghost box score 0.30~0.52 다수 발생

    조건 B (이동량 기반, 이전 프레임 존재 시):
      - shift > 20px → CRITICAL
        # vehicle_04 epsilon=0.01: 중심 좌표 ~387px→98px (급격 이동)
      - shift > 10px → RISK
        # epsilon=0.01 평균 오차 12.9px → threshold 직접 도출

    두 조건 중 더 심각한 쪽을 반환한다.
    """
    level = CorruptionLevel.STABLE

    # 조건 A: 신뢰도 점수
    if current.score < _SCORE_THRESHOLD_CRITICAL:
        level = CorruptionLevel.CRITICAL
    elif current.score < _SCORE_THRESHOLD_RISK:
        level = CorruptionLevel.RISK

    # 조건 B: 연속 프레임 이동량
    if previous is not None:
        shift = center_shift(previous.center, current.center)
        if shift > _SHIFT_THRESHOLD_CRITICAL:
            # 더 심각한 수준으로 격상
            level = CorruptionLevel.CRITICAL
        elif shift > _SHIFT_THRESHOLD_RISK and level is CorruptionLevel.STABLE:
            # epsilon=0.01 이상에서 평균 12.9px 오차 → RISK 격상
            level = CorruptionLevel.RISK

    return level


def is_corrupted(
    current: Detection,
    previous: Optional[Detection] = None,
) -> bool:
    """CE 여부를 반환한다 (RISK 이상이면 오염 판정)."""
    return assess_corruption(current, previous) is not CorruptionLevel.STABLE


# ══════════════════════════════════════════════════════════════════════════
# 2. 개선된 DDM 분류 함수
# ══════════════════════════════════════════════════════════════════════════

# DDM 리플레이 메모리 예치 비율 (논문 기준값 + CE 확장)
DDM_DEPOSIT_RATIO: dict[ExperienceType, float] = {
    ExperienceType.RE: 1.00,   # 결과 경험: 전량 저장
    ExperienceType.DE: 0.80,   # 위험 경험: 80% 저장
    ExperienceType.SE: 0.20,   # 안전 경험: 20% 저장
    ExperienceType.CE: 0.00,   # 오염 경험: 저장 차단 (pCE = 0.0)
    # pCE=0.0 근거: epsilon=0.01에서 D'OtoU 오차율 0.18%, θ 오차 0.17°
    # → 소량이지만 정책 학습에 편향을 줄 수 있어 완전 차단
}


def classify_experience(
    experience: Experience,
    current_det: Detection,
    previous_det: Optional[Detection],
    dist_to_obstacle: float,
    done: bool,
    danger_radius: float = 50.0,
) -> ExperienceType:
    """경험을 RE / DE / SE / CE로 분류한다.

    우선순위: CE > RE > DE > SE

    CE 판정 기준:
      - score < 0.70: epsilon=0.01 이상에서 ghost box 신뢰도 하락 확인
      - shift > 10px: epsilon=0.01 평균 오차 12.9px → 10px을 감지 경계로 설정

    Args:
        experience:        분류할 경험 객체
        current_det:       현재 프레임 검출 결과
        previous_det:      이전 프레임 검출 결과 (없으면 None)
        dist_to_obstacle:  UAV-장애물 거리 (픽셀 또는 스케일 단위)
        done:              에피소드 종료 여부 (목표 도달 / 충돌)
        danger_radius:     위험 판정 거리 임계값
    """
    # 1순위: 오염 감지 → CE
    if is_corrupted(current_det, previous_det):
        return ExperienceType.CE

    # 2순위: 에피소드 종료 (목표 도달 또는 충돌) → RE
    if done:
        return ExperienceType.RE

    # 3순위: 장애물 근접 → DE
    if dist_to_obstacle < danger_radius:
        return ExperienceType.DE

    # 기본: 안전 상태 → SE
    return ExperienceType.SE


# ══════════════════════════════════════════════════════════════════════════
# 3. 신뢰도 기반 보상 함수
# ══════════════════════════════════════════════════════════════════════════

# 보상 상수
_REWARD_STEP          =  0.10    # 안전 이동 시 스텝 보상
_REWARD_GOAL          = 10.00    # 목표 도달
_REWARD_COLLISION     = -10.00   # 충돌
_REWARD_APPROACH      = -0.10    # 장애물 접근 패널티 (기본)
_PENALTY_RISK         = -0.50    # RISK 오염 추가 패널티
# # epsilon=0.01~0.05 평균 θ 오차 0.17~0.38° → 방향 판단 오류 유발 수준
_PENALTY_CRITICAL     = -1.00    # CRITICAL 오염 추가 패널티
# # vehicle_04 ghost box 케이스: 2→8개 검출, 중심 오차 ~22.9px
# # D'OtoU 오차율 최대 1.29%(vehicle_04, epsilon=0.10)


def base_reward(
    dist_to_obstacle: float,
    goal_reached: bool,
    collision: bool,
    prev_dist: float = float("inf"),
) -> float:
    """오염 무관 기본 보상 함수 (FRDDM-DQN 원형).

    - 목표 도달: +10.0
    - 충돌:     -10.0
    - 장애물 접근 (거리 감소): 접근 패널티
    - 일반 이동: 스텝 보상
    """
    if goal_reached:
        return _REWARD_GOAL
    if collision:
        return _REWARD_COLLISION
    if dist_to_obstacle < prev_dist:
        return _REWARD_APPROACH
    return _REWARD_STEP


def compute_reward(
    dist_to_obstacle: float,
    goal_reached: bool,
    collision: bool,
    corruption: CorruptionLevel,
    prev_dist: float = float("inf"),
) -> float:
    """신뢰도 오염 수준을 반영한 보상 함수.

    STABLE  → 기존 보상 함수 그대로 적용
    RISK    → 장애물 접근 시 -0.5 추가 패널티
              # epsilon=0.01에서 12.9px 오차 → 거리 추정 신뢰도 저하
              # D'OtoU 오차율 0.18%, θ 오차 0.17° 수준에서 경미한 보정
    CRITICAL → -1.0 추가 패널티 (접근 여부 무관)
              # vehicle_04 epsilon=0.01: ghost box 8개, 중심 오차 22.9px
              # D'OtoU 오차율 최대 1.29% (vehicle_04, epsilon=0.10)
              # → 상태 벡터 자체를 신뢰할 수 없어 강한 억제 적용

    Args:
        dist_to_obstacle: 현재 UAV-장애물 거리
        goal_reached:     목표 도달 여부
        collision:        충돌 여부
        corruption:       오염 심각도
        prev_dist:        이전 스텝 거리 (접근 판단용)
    """
    r = base_reward(dist_to_obstacle, goal_reached, collision, prev_dist)

    if corruption is CorruptionLevel.STABLE:
        # 오염 없음: 기존 보상 유지
        # epsilon=0 기준 D'OtoU 오차율 0.0%, θ 오차 0.0°
        return r

    if corruption is CorruptionLevel.RISK:
        # 경미한 오염: 장애물 접근 시 추가 패널티
        # epsilon=0.01~0.05 평균 중심 오차 12.9~14.1px 구간
        if dist_to_obstacle < prev_dist:
            r += _PENALTY_RISK
        return r

    # CRITICAL: 심각한 오염 → 무조건 강한 억제
    # vehicle_04처럼 ghost box 폭증, 중심 오차 22px 이상 케이스
    r += _PENALTY_CRITICAL
    return r


# ══════════════════════════════════════════════════════════════════════════
# 4. 개선된 DDM 리플레이 버퍼
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ImprovedDDMBuffer:
    """CE를 저장에서 차단하는 DDM 우선 리플레이 메모리.

    각 경험 타입별 서브큐를 분리하여 샘플링 비율을 독립 제어한다.
    CE(pCE=0.0)는 add() 호출 시 즉시 폐기한다.

    샘플링 비율 (기본):
      RE : DE : SE = 4 : 3 : 3   (CE는 항상 0)
    """
    capacity: int = 10_000

    _buffers: dict[ExperienceType, deque] = field(init=False)
    _stats:   dict[ExperienceType, int]   = field(init=False)

    # CE 폐기 카운터 – 공격 탐지 빈도 모니터링용
    discarded_ce: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        cap = self.capacity
        self._buffers = {
            ExperienceType.RE: deque(maxlen=cap // 2),
            ExperienceType.DE: deque(maxlen=cap // 4),
            ExperienceType.SE: deque(maxlen=cap // 4),
            # CE 버퍼 없음 (pCE = 0.0)
        }
        self._stats = {t: 0 for t in ExperienceType}

    def add(self, exp: Experience) -> bool:
        """경험을 버퍼에 추가한다.

        CE는 DDM_DEPOSIT_RATIO에 따라 pCE=0.0이므로 항상 폐기.
        다른 타입은 확률적 예치 비율을 적용한다.

        Returns:
            True: 저장됨, False: 폐기됨
        """
        self._stats[exp.exp_type] = self._stats.get(exp.exp_type, 0) + 1
        ratio = DDM_DEPOSIT_RATIO[exp.exp_type]

        if ratio == 0.0 or random.random() > ratio:
            if exp.exp_type is ExperienceType.CE:
                self.discarded_ce += 1
            return False

        self._buffers[exp.exp_type].append(exp)
        return True

    def sample(self, batch_size: int) -> list[Experience]:
        """RE:DE:SE = 4:3:3 비율로 배치를 샘플링한다."""
        weights = {
            ExperienceType.RE: 4,
            ExperienceType.DE: 3,
            ExperienceType.SE: 3,
        }
        batch: list[Experience] = []
        total = sum(weights.values())

        for exp_type, w in weights.items():
            buf = self._buffers[exp_type]
            n   = max(1, round(batch_size * w / total))
            n   = min(n, len(buf))
            batch.extend(random.sample(list(buf), n))

        random.shuffle(batch)
        return batch[:batch_size]

    @property
    def size(self) -> int:
        return sum(len(b) for b in self._buffers.values())

    def stats_summary(self) -> str:
        lines = ["[DDM Buffer Stats]"]
        for t in ExperienceType:
            stored = len(self._buffers[t]) if t is not ExperienceType.CE else 0
            seen   = self._stats.get(t, 0)
            ratio  = DDM_DEPOSIT_RATIO[t]
            lines.append(
                f"  {t.name}: seen={seen:5d}  stored={stored:5d}"
                f"  deposit_ratio={ratio:.2f}"
            )
        lines.append(f"  CE discarded total: {self.discarded_ce}")
        lines.append(f"  Buffer size: {self.size}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# 동작 확인 (직접 실행 시)
# ══════════════════════════════════════════════════════════════════════════

def _demo() -> None:
    """실험 결과 수치를 이용한 동작 검증 데모."""
    print("=" * 60)
    print("개선된 DDM 동작 검증")
    print("=" * 60)

    # --- 오염 감지 테스트 ---
    clean_det = Detection(box=[231.2, 216.6, 400.8, 299.1],
                          score=0.9417, center=(316.0, 257.8))

    # epsilon=0.01에서 실제 관측된 ghost box (vehicle_04)
    # 중심 좌표가 (381, 95) → (98, 480) 으로 이동 (~384px shift)
    ghost_det = Detection(box=[6.9, 464.6, 190.8, 496.5],
                          score=0.5137, center=(98.8, 480.5))

    # epsilon=0.05에서 vehicle_01 주 박스 (중심 미세 이동)
    mild_det  = Detection(box=[236.3, 212.5, 396.1, 301.2],
                          score=0.9183, center=(316.2, 256.8))

    cases = [
        ("clean      (eps=0)   ", clean_det, None),
        ("ghost_box  (eps=0.01)", ghost_det, clean_det),
        ("mild_shift (eps=0.05)", mild_det,  clean_det),
    ]

    print("\n[1] 오염 감지 테스트")
    print(f"  {'케이스':<30} {'오염수준':<12} {'is_corrupted'}")
    print("  " + "-" * 54)
    for label, cur, prev in cases:
        lvl = assess_corruption(cur, prev)
        print(f"  {label:<30} {lvl.name:<12} {is_corrupted(cur, prev)}")

    # --- 보상 함수 테스트 ---
    print("\n[2] 보상 함수 테스트 (장애물 접근 상황)")
    print(f"  {'오염수준':<12} {'보상':>8}   설명")
    print("  " + "-" * 44)
    reward_cases = [
        (CorruptionLevel.STABLE,   "기존 보상 유지"),
        (CorruptionLevel.RISK,     "페널티 -0.5 추가"),
        (CorruptionLevel.CRITICAL, "페널티 -1.0 추가"),
    ]
    for lvl, desc in reward_cases:
        r = compute_reward(
            dist_to_obstacle=30.0,
            goal_reached=False,
            collision=False,
            corruption=lvl,
            prev_dist=35.0,        # 접근 중
        )
        print(f"  {lvl.name:<12} {r:>8.2f}   {desc}")

    # --- DDM 버퍼 테스트 ---
    print("\n[3] DDM 버퍼 저장 테스트")
    buf = ImprovedDDMBuffer(capacity=1000)
    dummy_state = [0.0] * 8

    # 각 타입별 100개씩 추가
    for exp_type in ExperienceType:
        for _ in range(100):
            exp = Experience(
                state=dummy_state, action=0, reward=0.0,
                next_state=dummy_state, done=False,
                exp_type=exp_type,
                corruption=(CorruptionLevel.CRITICAL
                             if exp_type is ExperienceType.CE
                             else CorruptionLevel.STABLE),
            )
            buf.add(exp)

    print(buf.stats_summary())

    # --- DDM 분류 테스트 ---
    print("\n[4] 경험 분류 테스트")
    classify_cases = [
        ("오염 검출 (ghost box)", ghost_det, clean_det, 30.0, False),
        ("충돌 종료",             clean_det, None,      5.0,  True),
        ("장애물 근접",           clean_det, None,      40.0, False),
        ("안전 이동",             clean_det, None,      200.0, False),
    ]
    dummy_exp = Experience(dummy_state, 0, 0.0, dummy_state, False)
    print(f"  {'상황':<22} {'분류'}")
    print("  " + "-" * 32)
    for desc, cur, prev, dist, done in classify_cases:
        result = classify_experience(dummy_exp, cur, prev, dist, done)
        print(f"  {desc:<22} {result.name}")


if __name__ == "__main__":
    _demo()
