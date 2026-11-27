# 9영역 Q·환경 검수 번들 v1·v2

구현: [EnvironmentReview](../../src/quality_estimation/environment_review.py), [선별·보정 연산](../../src/quality_estimation/environment_selection.py), [RGB 단독 재생](../../src/quality_estimation/infer_versioned_review.py). 이 계약은 `environment-quality-review-bundle-1`과 명시적 구간 보정을 추가한 `environment-quality-review-bundle-2`를 설명한다. 기존 `roi9-versioned-review-bundle-1`의3/4과제 GP 번들과 이전 X56 경로는 각 원래 형식으로 읽는다. 파일명은 기존 링크를 보존하기 위해 유지한다.

## 입력과 버전

인지 입력은 고정 전방 RGB와 해당 버전의 TwinLiteNet+ 출력이다. `SpatialProfile`의9영역·카메라 크기와 mask SHA를 공유한다. 품질 특징과 환경 특징의 순서를 번들에서 각각 지정하고, 겹치는 특징은 한 번 계산할 수 있다. **온라인 입력에는 GT·IoU·시뮬레이터 날씨/시각이 필요하지 않다.**

인지 체크포인트 SHA, baseline/adaptation identity, EMA/model state 종류를 정확히 연결한다. 품질 GP/RF·보정·정책이 다른 인지 버전에 속하면 로딩을 거부한다. RF와 RGB 분류기는 pickle 대신 배열을 저장해40/50 프로필에서 읽는다. 환경 분류기는 인지 가중치와 독립이지만 같은 카메라·특징 규약을 사용한다.

| 필수 파일 | 의미 |
|---|---|
| `bundle_manifest.json` | 형식, 인지 identity, 가중치 종류, ROI SHA, 특징 순서, 파일 SHA |
| `quality_model.npz` | 주 Q 하나를 예측하는 GP 또는 RF |
| `weather.npz`, `lighting.npz` | RGB만 사용하는 감독 분류기 |
| `centers.npz` | 학습된 RGB 분류 확률 표현의 e-means 중심·활성 차원 |
| `calibration.json` | 해당 인지·품질 버전의 Cal 부호 잔차 분포와 v2 구간 경계·선정 근거 SHA |
| `policy.json` | 공통/환경별 참조, 균형 할당 비중, 무작위 감사, 불확실성 탐색 비중 |

## Q와 보정

주 정답은 패딩을 제외한 전체 화면 white/yellow 차선 합집합의 IoU다. stopline은 주 Q에서 제외한다. TP+FP+FN=0이면 미정의이며0이나1로 채우지 않는다. 근거리·원거리 영상 띠의 GT precision/recall은 **오프라인 진단**이다. 이 형식은 ROI별 온라인 IoU나 별도의 근·원거리 성능 예측을 요구하지 않는다.

GP는 평균과 관측 분산을 내보낸다. 잡음 분산을 뺀 잠재 표준편차도 진단용으로 제공한다. RF의 트리 간 차이는 GP 불확실성으로 해석하지 않는다. RF에서는 `gp_uncertainty_available=false`이고, 잔차 단위 계산에 쓰는 scale1은 예측 표준편차1을 뜻하지 않는다.

Cal에서 `(y-mean)/scale`의 부호 잔차 분포를 고정한다. GP의 scale은 관측 표준편차, RF는1이다. 저IoU 검수 점수는 이 경험적 CDF로 계산한다. GP 표준편차가 크다는 것만으로 저IoU라고 판단하지 않으며, 오차·야간 포함률·구간 폭을 별도로 검증한다.

v1의 구간은 전체 Cal 잔차의5·95분위수로 계산한다. v2는 `interval_method`와 길이2의 유한·오름차순 `interval_residual_bounds`를 필수로 저장한다. `pooled`와 `road_envelope` 두 후보 중, Cal 도로 제외 교차검증과 primary Dev의 평균90% interval score가0.005 이상 개선되고 Dev 포함률이 떨어지지 않을 때 후자를 선택한다. score는 `폭 + 20×구간 밖 거리`이므로 무조건 넓히는 방식에 불이익을 준다. `road_envelope`는 충분한 표본을 가진 Cal 도로별5분위수의 최솟값·95분위수의 최댓값이다. 실행 시에는 두 고정 경계만 사용하며 도로 ID나 GT를 요구하지 않는다. 출력 구간은 `[mean + lower×scale, mean + upper×scale]`를 `[0,1]`로 자른 값이다.

v2 구간 보정은 점 추정·표준편차·검수 CDF·순위를 변경하지 않는다. 따라서 구간 경계가 검수 CDF의5·95분위수와 다를 수 있다. 이는 경험적 예측 구간이며 GP의 베이지안 credible interval이나 환경별90% 보장을 뜻하지 않는다. 최종 포함률과 폭은 미개봉 확인 자료에서 함께 보고한다. 인지 버전이 바뀌면 새 Cal·Dev에서 다시 선택하고, 기존 f0 번들·선별 은행은 진단 실험으로 덮어쓰지 않는다. v1 전용 구현이 v2를 조용히 오독하지 않도록 형식 번호를 올렸다.

## 환경과 검수

날씨 코드는 수집 SDK의 Sunny0/Cloudy1/Foggy2/Storm3/Rainy4/Snowy5다. 조명 라벨은 시각을 day0 `[08,18)`, transition1 `[06,08) ∪ [18,20)`, night2 나머지로 묶는다. RGB 분류 결과는 이러한 수집 라벨의 예측이며 물리적 가시성·운행 정상성의 정답은 아니다.

환경 표현은 RGB 분류기의 날씨6확률+조명3확률에 제곱근을 취한9차원이다. e-means는 이 표현을 군집화한다. **감독 RGB 표현 뒤의 군집화**로 명시하며 비지도 날씨 발견이라고 부르지 않는다. 온라인에서는 고정 중심에 최근접 할당한다.

정책은 `global`, `rgb_lighting`, `rgb_weather`, `rgb_tag`, `rgb_cluster`를 지원한다. 세분 그룹의 지원이 부족하면 공통0.7을 쓰고, 조명만/날씨만의 더 굵은 정책은 별도 Dev 비교로 고른다. 시뮬레이터 정답 그룹은 오프라인 대조이며 배포 정책으로 선택하지 않는다. RGB 저신뢰·중심 거리 초과 여부도 남긴다.

환경별 IoU 참조는 **검수 우선순위**만 바꾼다. 비교용 `IoU<0.7`의 의미나 가시성 정상 라벨을 바꾸지 않는다. 균형 슬롯 일부를 먼저 예약하고 남은 슬롯은 전체 위험 순위로 채운다. 무작위 감사는 `floor(예산×0.1)`장을 나머지에서 중복 없이 뽑는다. 불확실성 순위는 추론 미니배치가 아니라 닫힌 검수 버퍼 전체에서 계산한다.

선별 precision은 저IoU 정답 수/전체 검수 슬롯 수다. 미정의 IoU를 제외한 precision과 미정의 장수도 따로 보고한다. GT 연결 후에도 개발·보정·확인 역할을 보존하며, 검수 목록 생성이 곧 학습 편입 승인은 아니다. 인지 가중치가 바뀌면 새 출력으로 X·y·품질모델·잔차 분포·환경 참조를 다시 만든다.
