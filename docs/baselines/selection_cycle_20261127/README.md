# 군집·검수·동일 예산 재학습 근거

[통합 결과와 채택 판정](../../experiments/군집_검수_동일예산_재학습_20261127.md)의 공개 집계·그림이다. 원본RGB/Semantic·체크포인트·개별ID·위치·상세 로그는 포함하지 않는다. 실제 실행물은 내부 실험 볼륨에서 같은 해시로 인계한다.

**최종 상태는 후보 보존·f0/g0 유지다.** 새 인지 후보는 개선됐지만 g1 MAE 유지 기준과 군집/σ혼합 검수 기준을 실패했다. 표의 새f1/g1 성적은 운영 채택 결과로 읽지 않는다.

## 집계 읽는 순서

| 파일 | 내용 |
|---|---|
| [runtime_evidence.json](runtime_evidence.json) | 인지/품질 쌍 채택 여부, 실제 검수 정책, 모델·번들 해시 |
| [feature_roles.json](feature_roles.json) | 실제 선정된 품질 열·보조 목표·ROI·자료 수 |
| [clustering_comparison.json](clustering_comparison.json) | RGB 표현·K·시드 비교와 선택 |
| [environment_diagnostics.json](environment_diagnostics.json) | 날씨/시각/위치 NMI, 동일 초기화 e-means/Lloyd 대조 |
| [environment_diagnostics_role_context.json](environment_diagnostics_role_context.json) | 군집 진단의 넓은 원장과 최종 품질 역할별 모집단 구분 |
| [admission_summary.json](admission_summary.json) | 방법별3,072장의 환경·군집 구성 |
| [training_budget.json](training_budget.json) |8회 학습의 실제 시간·메모리·해시, 중복/정답 보존 검사 |
| [perception_comparison.json](perception_comparison.json) | 신규/기존 인지 성능, 두 시드, 원시/공통계수/선택계수 비교 |
| [per_road_quality_descriptive.json](per_road_quality_descriptive.json) | 도로별오차·편향·정답분모의 사후 기술통계 |
| [quality_comparison.json](quality_comparison.json) | GP/RF·확률·공간 진단·동일 예산 검수 |
| [quality_bo.json](quality_bo.json) · [rf_quality_bo.json](rf_quality_bo.json) | 같은 쿼리 수의 BO/무작위 품질모델 튜닝 |
| [policy_bo.json](policy_bo.json) | 검수 정책 탐색과 개발 점수 |
| [environment_reference_f1.json](environment_reference_f1.json) | 새 인지 버전의 환경별 상대 기준 진단 |
| [acceptance_gates.json](acceptance_gates.json) · [analysis_extensions_frozen.json](analysis_extensions_frozen.json) | 고정 채택 기준과 사전 추가 분석 규칙 |
| [per_road_quality_descriptive.json](per_road_quality_descriptive.json) | 역할별 도로 오차·편향의 사후 진단(재튜닝 없음) |
| [verification_followup.json](verification_followup.json) | 배치 구성 차이·인계 위치 변경·최종 테스트와 후속 코드 근거 |
| [runtime_verification.json](runtime_verification.json) | RGB단독·GT연결·40CPU/50GPU·RF 포터블 검증 |
| [complete_study_verification.json](complete_study_verification.json) |8학습·24파생 가중치·로그·입력·평가 카운트 검증 |
| [evidence_inventory.json](evidence_inventory.json) | 공개 파일별SHA-256 |

## 발표·페이지용 그림

PNG는1920×1080이다. 개인 실험 볼륨에는 같은 그래프의SVG와 사례별640×360 개별 패널도 보관한다. 슬라이드에서는 비율을 유지하고 작은 글자가 읽히는지 확인한다. 합성 이미지의 GT는 실제 Semantic 정답이며 색상 범례를 함께 넣었다.

| 그림 | 용도 |
|---|---|
| [01_versioned_pipeline.png](figures/01_versioned_pipeline.png) | 전체 흐름과 오프라인/실행 역할 |
| [02_roi_and_feature_roles.png](figures/02_roi_and_feature_roles.png) |9영역·품질/환경 입력의 구분 |
| [03_environment_recipe.png](figures/03_environment_recipe.png) | 군집 표현과 K/시드 선택 |
| [04_admission_environment_coverage.png](figures/04_admission_environment_coverage.png) | 선별 방법별 환경 구성 |
| [05_environment_representatives.png](figures/05_environment_representatives.png) | 실제 환경 외관의 군집 대표 |

[인지 개선과 유지](figures/06_retraining_and_retention.png) · [보간 trade-off](figures/07_interpolation_tradeoff.png) · [GP/RF 오차](figures/08_quality_gp_rf.png) · [검수](figures/09_review_budget.png) · [불확실성과 포함률](figures/10_uncertainty_and_coverage.png) · [동일계수 선별](figures/11_matched_selection.png) · [공분산 대조](figures/12_task_covariance.png)

[가장 큰 개선](figures/case_A_maximum_change.png) · [가장 큰 감소](figures/case_C_minimum_change.png) · [큰 품질 추정 오차](figures/case_D_largest_prediction_error.png). 사진 한 장의 개선을 전체 효과로 대신하지 않는다.

## 재현 경계

공개 재현·해석 코드 기준은 `1fe4a67`이다. 각 학습의 실제 소스 스냅샷·해시는 실행별 원장에 보존했다. 배치 재현 검사와 그림 설명의 후속 코드는 `b952fa8`이다. [실험 실행 순서](../../../scripts/experiments/selection_cycle_20261126/README.md), [현재 특징/버전 실행](../../guides/ROI와_특징추출.md#버전-쌍의-현재-실행), [모델 갱신 절차](../../guides/인지모델_재학습과_갱신판정.md)를 따른다. 실행에는 정확한 인지 adaptation 디렉토리와 짝 review bundle 전체가 필요하다. public JSON만으로 원본이나 가중치를 복원할 수 없다.

8회는 optimizer 학습,24개는 추가 학습 없이 만든 보간 가중치다. 두 시드는 같은 선별 bank를 사용한다. 이미 열었던 f0의 도로를 재학습 이후 paired 진단으로 사용했고 모두 같은 지도에 있으므로 처음 보는 독립 최종 시험이나 보편적 성능을 주장하지 않는다. Q<0.7은 공통 연구 벤치마크이며 운행 안전·야간 정상 기준은 별도다.
