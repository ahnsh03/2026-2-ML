# 지원 오류를 구분하는 버전별 검수 번들 v1

형식 식별자는 `support-aware-review-bundle-1`이다. [현재 실험](../experiments/표시가용성_누적갱신_20261128.md)의 오프라인 후보 형식이며 기존 `environment-quality-review-bundle-1/2`를 변경하지 않는다. 형식 구현과 오프라인 채택은 별도다.

## 입력과 신원

- 고정 전방 카메라1280×720, `roi9-front-v1`, 동결 ROI 해시와 명시적인 특징 순서가 필요하다.
- Q 회귀와 오류 회귀는 서로 다른 특징 부분집합·RF/GP 계열을 사용할 수 있다. 입력 계산은 두 열 목록의 순서 보존 합집합을 한 번 추출한다. 같은 과제의 RF/GP는 같은 행과 열로 비교한다.
- 환경 입력은 RGB 히스토그램400열+보닛/상부/전체의 광도 요약8열이다. 이408열을 Q/오류 모델에 모두 넣는다는 의미가 아니다.
- 번들은 체크포인트 SHA, 원래v6 SHA, 타깃 정책 SHA, adaptation run SHA와 선택한 `ema_state_dict`/`model_state_dict`를 묶는다. Q/오류 모델·Cal·정책·환경 파일과 모든 해시를 확인한다. 다른 버전은 같은 특징 수라도 혼용하지 않는다.
- 현재 빌더는 명시적인 재학습 인지 버전의 `model_state_dict` 팩을 받는다. 새 가중치에는 X/y·Q/오류 회귀·잔차·정책을 다시 만든다. RGB 환경 모델은 원본 RGB/역할/ROI가 동일하고 별도 검증됐을 때 재사용할 수 있다.

## 오프라인 목표

Q는 패딩을 제외한 white/yellow 차선 합집합 IoU다. union0은 미정의로 보존한다. stopline은 Q에 넣지 않지만 인지모델 유지 평가에는 포함한다.

별도 오류 목표는 근거리 FN 또는 FP, 원거리 FN 또는 FP, 도로 밖 FP의3개 지시값이다. native640×360에서 GT≥50픽셀이고 recall<0.8이면 FN, 예측≥50픽셀이고 precision<0.8이면 FP다. 근거가 없는0을 안전/정상으로 읽지 않는다. 실제 무출력 FN과 GT가 없는 곳의 FP도 정의할 수 있다. 기준값과 가중치는 실험 전에 고정한 공학적 설정이다.

검수 가치는 `near + 0.25*far + 0.5*offroad_fp`이고 범위는0–1.75다. GP는 이항 지시값에 대한 Gaussian 회귀 근사이며 출력 점수는 보정된 실패 확률이 아니다. GP 선형 결합의 평균·잠재 공분산을 그대로 전파한다. 표시 유무의 GT와 ROI별 IoU는 온라인 입력에 요구하지 않는다.

## 출력과 선별

| 필드 | 의미 |
|---|---|
| `estimated_iou` | 전체 Q 평균,0–1로 제한 |
| `interval90_lower/upper` | Cal 부호 잔차로 만든 명목90% 경험 구간; 환경별 포함률 보장은 없음 |
| `quality_predictive_scale` | Q-GP의 관측 표준편차 또는 RF의 단위 scale |
| `low_quality_probability` | 고정 Q0.7에 대한 경험적 Cal CDF 점수 |
| `error_task_scores` | 표시용으로0–1에 제한한3오류 회귀 점수; 이항 확률로 해석하지 않음 |
| `utility_raw_mean`, `utility_latent_sd` | 제한하지 않은 오류 GP/RF의 선형 검수 가치와 GP 잠재 SD; RF의 SD는0이며 가용성 필드로 구분 |
| `near_minus_far_error_score/latent_sd` | 근거리−원거리 오류 점수의 Gaussian 대비; 보정된 공간 실패 판정이 아님 |
| `review_score_is_probability` | `quality_probability` 모드에서만 true. `supported_utility`는 false |
| `review_threshold_used` | 환경별 Q 참조가 실제 점수에 쓰였는지 명시. 직접 오류 가치 모드에서는 false |
| `environment_cluster`, `environment_ood` | RGB 중심 군집과 저신뢰/거리 기반 감사 표시; OOD 보장 확률은 아님 |
| `expected_near_marking_score` | RGB 표시 맥락 분류 점수. 이 값만으로 무출력 후보를 제외하지 않음 |

선별은 닫힌 전체 버퍼에서 수행하며 미니배치마다 예산/순위를 재정의하지 않는다. 오류 가치 우선, 환경 예약, 정보 검수, 무작위 감사를 구분한다. 조건부 분산 검수는 먼저 고른 프레임에 대해 추가 라벨을 관측한다고 가정한 공분산 감소로 중복을 줄인다. 실제 라벨값이나 온라인 ROI IoU를 읽지 않는다. 후보의 조건부 포함 확률은 각 선택 단계의 값이며 전체 모집단의 무조건 확률을 뜻하지 않는다.

## 실행·GT 연결·인계

[빌더](../../scripts/experiments/scene_cycle_20261128/build_support_bundle.py)는 새 출력 폴더에만 쓰고 완성된 후보를 덮어쓰지 않는다. [RGB 재생](../../src/quality_estimation/infer_versioned_review.py)은 GT 없이 입력→특징→추정→고정 예산 후보를 만든다. [GT 연결](../../src/quality_estimation/join_spatial_gt.py)은 이후 원본 RGB/Semantic 바이트·픽셀·버전·ROI·팩 신원을 검사하고 `supported_error_targets`를 추가한다. 확인·보정·개발 역할은 해당 주기에서 학습 편입을 차단하며 `admission_approved=False`를 유지한다.

[인계 패키저](../../scripts/package_environment_release.py)는 코드·상위 모델 코드·인지/검수 아티팩트·RGB예제를 묶으며 원본GT 이미지를 포함하지 않는다. [독립 실행 검사](../../scripts/verify_environment_release_runtime.py)는 배포 볼륨만 마운트하여40프로파일 CPU/50프로파일 실제GPU를 비교한다. 충분한 검수 예산을 지정하면 정보/무작위 슬롯도 검사할 수 있다. 실제40시리즈 GPU는 별도 장비 검증이 필요하다. 이 실행 검사는 온라인 ROS 연결이나 오프라인 채택을 승인하지 않는다.
