# 9영역 Q·환경 검수 번들 v1

구현: [EnvironmentReview](../../src/quality_estimation/environment_review.py), [선별·보정 연산](../../src/quality_estimation/environment_selection.py), [RGB 단독 재생](../../src/quality_estimation/infer_versioned_review.py). 이 계약은 새 `environment-quality-review-bundle-1` 형식이다. 기존 `roi9-versioned-review-bundle-1`의3/4과제 GP 번들과 이전 X56 경로는 각 원래 형식으로 읽는다.

## 입력과 버전

인지 입력은 고정 전방 RGB와 해당 버전의 TwinLiteNet+ 출력이다. `SpatialProfile`의9영역·카메라 크기와 mask SHA를 공유한다. 품질 특징과 환경 특징의 순서를 번들에서 각각 지정하고, 겹치는 특징은 한 번 계산할 수 있다. **온라인 입력에는 GT·IoU·시뮬레이터 날씨/시각이 필요하지 않다.**

인지 체크포인트 SHA, baseline/adaptation identity, EMA/model state 종류를 정확히 연결한다. 품질 GP/RF·보정·정책이 다른 인지 버전에 속하면 로딩을 거부한다. RF와 RGB 분류기는 pickle 대신 배열을 저장해40/50 프로필에서 읽는다. 환경 분류기는 인지 가중치와 독립이지만 같은 카메라·특징 규약을 사용한다.

| 필수 파일 | 의미 |
|---|---|
| `bundle_manifest.json` | 형식, 인지 identity, 가중치 종류, ROI SHA, 특징 순서, 파일 SHA |
| `quality_model.npz` | 주 Q 하나를 예측하는 GP 또는 RF |
| `weather.npz`, `lighting.npz` | RGB만 사용하는 감독 분류기 |
| `centers.npz` | 학습된 RGB 분류 확률 표현의 e-means 중심·활성 차원 |
| `calibration.json` | 해당 인지·품질 버전에서 얻은 Cal 부호 잔차 분포 |
| `policy.json` | 공통/환경별 참조, 균형 할당 비중, 무작위 감사, 불확실성 탐색 비중 |

## Q와 보정

주 정답은 패딩을 제외한 전체 화면 white/yellow 차선 합집합의 IoU다. stopline은 주 Q에서 제외한다. TP+FP+FN=0이면 미정의이며0이나1로 채우지 않는다. 근거리·원거리 영상 띠의 GT precision/recall은 **오프라인 진단**이다. 이 형식은 ROI별 온라인 IoU나 별도의 근·원거리 성능 예측을 요구하지 않는다.

GP는 평균과 관측 분산을 내보낸다. 잡음 분산을 뺀 잠재 표준편차도 진단용으로 제공한다. RF의 트리 간 차이는 GP 불확실성으로 해석하지 않는다. RF에서는 `gp_uncertainty_available=false`이고, 잔차 단위 계산에 쓰는 scale1은 예측 표준편차1을 뜻하지 않는다.

Cal에서 `(y-mean)/scale`의 부호 잔차 분포를 고정한다. GP의 scale은 관측 표준편차, RF는1이다. 이 경험적 분포로 저IoU 검수 점수와90% 구간을 만든다. 실제 구간 포함률은 독립 확인 자료에서 측정하며, 환경별 포함률이나 독립 표본 보장을 가정하지 않는다.

## 환경과 검수

날씨 코드는 수집 SDK의 Sunny0/Cloudy1/Foggy2/Storm3/Rainy4/Snowy5다. 조명 라벨은 시각을 day0 `[08,18)`, transition1 `[06,08) ∪ [18,20)`, night2 나머지로 묶는다. RGB 분류 결과는 이러한 수집 라벨의 예측이며 물리적 가시성·운행 정상성의 정답은 아니다.

환경 표현은 RGB 분류기의 날씨6확률+조명3확률에 제곱근을 취한9차원이다. e-means는 이 표현을 군집화한다. **감독 RGB 표현 뒤의 군집화**로 명시하며 비지도 날씨 발견이라고 부르지 않는다. 온라인에서는 고정 중심에 최근접 할당한다.

정책은 `global`, `rgb_lighting`, `rgb_weather`, `rgb_tag`, `rgb_cluster`를 지원한다. 세분 그룹의 지원이 부족하면 공통0.7을 쓰고, 조명만/날씨만의 더 굵은 정책은 별도 Dev 비교로 고른다. 시뮬레이터 정답 그룹은 오프라인 대조이며 배포 정책으로 선택하지 않는다. RGB 저신뢰·중심 거리 초과 여부도 남긴다.

환경별 IoU 참조는 **검수 우선순위**만 바꾼다. 비교용 `IoU<0.7`의 의미나 가시성 정상 라벨을 바꾸지 않는다. 균형 슬롯 일부를 먼저 예약하고 남은 슬롯은 전체 위험 순위로 채운다. 무작위 감사는 `floor(예산×0.1)`장을 나머지에서 중복 없이 뽑는다. 불확실성 순위는 추론 미니배치가 아니라 닫힌 검수 버퍼 전체에서 계산한다.

선별 precision은 저IoU 정답 수/전체 검수 슬롯 수다. 미정의 IoU를 제외한 precision과 미정의 장수도 따로 보고한다. GT 연결 후에도 개발·보정·확인 역할을 보존하며, 검수 목록 생성이 곧 학습 편입 승인은 아니다. 인지 가중치가 바뀌면 새 출력으로 X·y·품질모델·잔차 분포·환경 참조를 다시 만든다.
