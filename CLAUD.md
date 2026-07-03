# FRDDM-DQN 적대적 공격 분석 프로젝트

## 프로젝트 목적
FRDDM-DQN 논문(2024)의 Faster R-CNN 인식 모듈에
적대적 예제 공격(FGSM)을 적용하여 바운딩 박스 오차를
측정하고, 그 오차가 DQN 상태 벡터에 미치는 영향을 분석한다.

## 기술 스택
- Python 3.11
- PyTorch 2.12.0, torchvision
- Faster R-CNN (fasterrcnn_resnet50_fpn)

## 핵심 수식 (논문 기반)
- 바운딩 박스 중심 좌표 변환 (수식 15)
- 상대 좌표 변환 (수식 16)
- D'OtoU 계산 (수식 18)
- theta'_o 계산 (수식 19)

## 실험 목표
- epsilon 0.01 ~ 0.10 범위에서 FGSM 공격 강도별
  바운딩 박스 좌표 오차 측정
- 측정된 오차를 상태 벡터 왜곡으로 변환하여 시각화

## 폴더 구조
- data/images/: 학습 및 테스트 이미지
- src/: 실험 코드
- results/: 실험 결과 그래프 및 수치