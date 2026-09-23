# B1. 휠 슬립·stuck 검출

> 계열 B (ERC) · **순위 3 — B 계열 중 최우선**
>
> **9/23 판정:** [C1 DR 오차·불확실성](./C1_DR오차_불확실성.md)에 흡수(슬립 = 운동 모델 잔차). 로그에서 stuck을 찾는 부분은 [B4](./B4_주행로그_군집.md) 거동 군집이 다룬다. 물리 실험 GT가 필요해 새 기준(낮은 개발 부담)에서는 순위 밖.

## 한 줄 정의

> **"이 로버는 지금 헛바퀴를 돌고 있는가"**
> 명령·IMU·GPS로부터 슬립(traction loss)과 stuck 상태를 실시간 판별한다.

## 동기

ERC 공식 트랙 명세가 Off-road 트랙의 핵심 과제로 **`traction loss`, `전복/stuck 위험`** 을 명시한다.
Marathon 트랙(Urban→Indoor→Off-road 연속)에서는 도메인 전환 시 실패 모드 관리가 핵심이다.
슬립을 즉시 알면 복구 동작을 트리거하거나 경로를 바꿀 수 있다.

## 데이터

| 항목 | 내용 |
|---|---|
| FrodoBots-2K | `data/output_rides_17.zip` — 303 rides (~35 GB) |
| Mini-4K | `data/mini4k/` — 마라톤 전 구간 약 32 h |
| 자체 실차 run | `data/runs/` ERRAWBIN 로그 (~1.9 GB) |
| 통제 실험 | `data/기본 거동 실험/` — stationary / straight / square / zigzag DR CSV |

### ride 로그 스키마 `[확정]`

```
control_data : linear, angular, rpm_1, rpm_2, rpm_3, rpm_4, timestamp
gps_data     : latitude, longitude, timestamp
imu_data     : compass, accelerometer, gyroscope, timestamp   ← 중첩 JSON 문자열
front/rear_camera_timestamps, mic_audio_timestamps
```

### 라벨 설계 (이 주제의 핵심 난점)

GT pose가 없으므로 라벨을 직접 만들어야 한다. 후보:

1. **명령 속도 vs GPS 유도 속도의 괴리** — `linear` 명령이 있는데 GPS 변위가 없으면 stuck
2. **휠 회전(rpm) vs 실제 이동거리 불일치** — 슬립 비율
3. **IMU 진동 특성** — 자갈·미끄러짐 시 고주파 성분 증가
4. `기본 거동 실험` CSV의 `err_imu` / `err_wheel` — gt 대비 오차가 이미 계산되어 있음

→ 1·2를 조합해 약한 라벨(weak label)을 만들고, 4의 통제 실험으로 검증하는 것이 현실적이다.

## 알고리즘

| 단계 | 모델 | 주차 |
|---|---|---|
| 0 | rpm/속도 비율 임계값 (베이스라인) | — |
| 1 | Logistic Regression / SVM | 3·7 |
| 2 | **Random Forest / Gradient Boosting** | 11 |
| 3 | GP — 슬립 정도 회귀 + 불확실성 | 12 |
| 분석 | PCA(IMU 스펙트럼 축약), **GMM/EM(주행 regime 군집: 직진·회전·슬립·정지)** | 6·13 |

## 강점

- **실차 노이즈 데이터.** "머신러닝다움"이 시뮬레이션보다 강하다.
- **동기가 자명하고 설명이 한 문장에 끝난다.** 딥러닝 배경 불필요.
- 작년 예시 `2LEOD` 계열이고 데이터 규모가 크다.
- 2주차 "데이터 전처리" 내용을 제대로 보여줄 소재 (중첩 JSON 파싱, 리샘플링, 시간 정렬, 아웃라이어).

## 약점 / 리스크

- **라벨 설계가 임의적일 수 있다.** 발표에서 "그 라벨이 맞다는 근거는?"이 첫 질문이 된다.
  → `기본 거동 실험` 통제 데이터로 라벨 정의를 검증하는 절을 반드시 넣을 것.
- **전처리 부담이 크다.** IMU CSV가 중첩 JSON 문자열, 센서별 Hz·타임스탬프 단위 불일치
  (control은 초 단위 float, gps는 밀리초 정수).
- **본선 로그 확보가 간접적이다.** 안승현은 출국하지 않는다 → 기존 데이터만으로 완결되게 설계할 것.

## 전환 조건

A1/A2가 [사전검증](../04_실행/9월_사전검증_체크리스트.md)에서 실패했을 때
(라벨 분산 부족 또는 베이스라인 AUC < 0.7) **B1으로 전환한다.**
