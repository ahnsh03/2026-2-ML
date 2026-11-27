# 환경 검증·전체 원본 재생: 표와 그림

**판정: 연구 후보 보존, 통합 교체·온라인 이행 보류.** 상세 해석은 [실험 기록](../../experiments/환경검증_전체원본재생_20261128.md), 실행은 [코드 안내](../../../scripts/experiments/environment_cycle_20261127/README.md)를 따른다. 이 폴더는 집계·재현 영수증·발표용 그림이며 원본 데이터와 가중치는 포함하지 않는다.

## 가장 먼저 읽을 결과

- 원본26,001장 전체를 공통 재생 풀로 보존했다. 같은 v6 출발·신규3,072장 예산의4방법과 원본 전용 대조를 합쳐8회 학습했다. 원래 날씨/시각이 없는 자료에 추정 태그를 정답으로 채우지 않았다.
- 새 확인618장의 GP 후보 도로평균 인지 IoU는0.19877→0.23650이었다. 다만 두 구간은 근거리 GT가 없고 한 표시 구간에서 실질적인 이득이 났다. **복수의 유효 표시 도로에서 개선을 입증한 결과로 쓰지 않는다.**
- 같은 alpha에서 평균은 GP 위험 선별0.23569, 무작위0.23831, 시뮬 태그 균형0.23923이다. **위험 선별이 무작위보다 재학습에 유리하다는 증거는 없다.** 반복은 같은 선별 목록의 optimizer 시드이며 목록 재추출 반복이 아니다.
- 품질 추정은 GP가 Dev에서 좋았으나 새 확인 MAE는 GP52=0.03653/RF52=0.03047로 역전됐다. 현재 GP 후보를 모든 조건의 최선으로 확정하지 않는다.
- 직접 RGB 날씨 balanced accuracy91.21%, 조명69.86%다. 07시 전이 라벨은17/156장만 맞혔다. 감독 분류 확률을 군집화한 것이며 e-means의 독자적인 환경 발견이나 Lloyd 대비 우월성을 입증하지 않았다.
- 고정한10% 검수62장 중59장이 낮은 IoU라 precision95.16%였지만 **근거리 결함114장을 한 장도 회수하지 못했다.** 59장이 근거리 진단 미지원 장면에 배정됐다. 무표시 구간의 실제 FP43장도 놓쳤다. 도로평균 recall9.54%는 무작위10.15%보다 낮아 사전 검수 기준을 실패했다.
- 동결한 BO 선정 정책은 시뮬 태그를 정확히 쓰는 대조도 검수 문제를 해결하지 못했다. 다만 RGB 날씨별 Dev 목적은 동일 예산 무작위 탐색0.37581이 BO0.39159보다 좋았다. 당시 BO 후보만 고르는 규칙이어서 무작위 최선은 운영 후보에 들지 않았다. **BO 우세나 환경별 기준 전체의 무효를 주장하지 않는다.** 다음은 새 개발 자료에서 합집합 선택·균형/임계 대조를 사전 고정한 뒤 새 독립 확인이다. 현재 확인에 임계를 다시 맞추지 않았다.

## 발표·페이지용 그림

PNG는 슬라이드 삽입용이며 같은 이름의 SVG가 있으면 확대/편집용으로 사용할 수 있다. 원본 비율을 유지한다. 그림의 지표는 전체 픽셀 pooled, 프레임 평균, 도로평균을 구분해서 읽는다. 그림08·15는 래스터 원본이며 SVG가 없다.

| 번호 | 파일 | 설명과 사용 시 주의 |
|---|---|---|
| 01 | [원본 구성 보존](figures/01_original_replay_composition.png) | 전체26,001장과 직전4,096장 부분 재생의 수집 묶음 비율. 과거 부분 재생을 완전 무작위라고 설명하지 않는다. |
| 02 | [개발 환경 분류](figures/02_environment_development.png) | RGB 표현과 군집 비교. Dev 성적이며 독립 확인은07/10과 함께 본다. |
| 03 | [신규 선별 환경 분포](figures/03_admission_environment_balance.png) | 신규 bank만의 분포.18그룹 중 부족한3그룹은 포화돼 완전 균등은 아니다. |
| 04 | [f0 품질 GP/RF](figures/04_f0_quality_regression.png) | 기존 인지 버전의 품질 회귀. 새 f1 성적과 섞지 않는다. |
| 05 | [재학습 Dev](figures/05_neural_development_raw.png) | 원시 가중치의 개발 성적. 보간과 새 확인의 모든 방법은09를 함께 본다. |
| 06 | [새 확인 인지·품질](figures/06_fresh_neural_and_quality.png) | 인지 개선과 새 품질모델 유지. 두 희소 표시 구간 때문에15·16의 지원량 설명이 필수다. |
| 07 | [새 확인 환경](figures/07_fresh_environment.png) | 날씨와 조명을 따로 평가. Dev1.0만 제시하지 않는다. |
| 08 | [RGB·실제 GT·전후 예측](figures/08_paired_gt_cases.png) | 평가 후 큰 야간·07시 개선과 최악 변화를 골랐다. **사후 극단 사례이며 대표 평균이 아니다.** 실제 GT를 원본 Semantic에서 다시 생성해 검증했다. |
| 09 | [동일 alpha·8학습 비교](figures/09_fresh_selection_comparison.png) | 무작위·균형·RF·GP·원본 전용. 점은 optimizer 시드, X는 Dev 유지/이득 실패다. |
| 10 | [날씨·조명 혼동행렬](figures/10_fresh_weather_confusion.png) | 직접 분류와 군집 해석,07시 오류 확인. 행별 정규화와 지원 장수를 함께 본다. |
| 11 | [환경별 정책 전이](figures/11_fresh_relative_policy_review.png) | 공통·시뮬 oracle·RGB·군집 임계의 고정 정책 비교. 확인을 보고 다시 선택한 정책이 아니다. |
| 12 | [조명별 인지 변화](figures/12_fresh_perception_lighting.png) | 같은 새 확인의 조명별 결과. 시뮬 시각 구간을 물리적 가시성 정답으로 부르지 않는다. |
| 13 | [구간 포함률과 폭](figures/13_interval_coverage_and_width.png) | 경험적90% 구간. 포함률만 높이려는 넓은 구간과 정확한 점 예측을 구분한다. |
| 14 | [환경 평균 대조](figures/14_quality_vs_environment_context.png) | 같은 fit 정답을 쓴 그룹 평균과 품질 GP. 단순 그룹 평균이 모든 RGB 단독 모델을 대표하지는 않는다. |
| 15 | [희소 표시 구간의 원본](figures/15_sparse_marking_support.png) | 세 구간 RGB·native Semantic·지원량. 흰/노랑 GT에는 방향 화살표 등 노면표시도 포함된다. |
| 16 | [검수 예산의 잘못된 배분](figures/16_review_support_failure.png) | 구간별 근거리 GT/실제 결함과 검수 장수. 높은 low-IoU precision과 유용한 실패 회수는 다르다. |

## 고정 설계·표·검증 영수증

| 질문 | 정본 |
|---|---|
| 최종 채택 여부·기준별 결과 | [전체 확인 판정](reports/fresh_confirmation_assessment.json), [후보 버전 쌍](deployment/candidate_pair.json) |
| 희소 표시와 근거리 FP/FN 원인 | [고정 선택 지원량 감사](reports/fresh_support_audit.json). GT는 선택 이후 연결했고 모델·정책·원래 채택 판정은 바꾸지 않았다. |
| 원본이 정말26,001장이었는가 | [학습 출처](reports/f0_training_provenance.json), [원본 구성](reports/original_replay_audit.json), [실제 sampler 노출](reports/original_sampler_exposure_audit.json), [정답 캐시 대조](reports/full_original_cache_audit.json) |
| 신규 목록과 누수·중복 | [선별/GT 연결 계보](reports/selection_lineage_no_candidate_gt.json), [선별 목록 겹침](reports/selection_bank_overlap.json), [동일 정책 은행](reports/identical_policy_banks.json) |
| GP/RF와 입력 열 | [f0 선택](f0_quality/quality_selection_frozen.json), [f1 선택](g1_quality/quality_selection_frozen.json) |
| e-means 표현과 대조 | [환경 표현 선택](environment/environment_selection_frozen.json), [동일 표현 군집 대조](reports/learned_representation_cluster_controls.json), [공식 구현 재현](reports/clustering_implementation_reproduction.json) |
| 환경별 임계·CDF 동점 | [g1 고정 정책](g1_quality/review_policy_frozen.json), [동점 대조](g1_quality/review_tie_study.json) |
| BO보다 좋은 무작위 후보의 누락 | [Dev 전용 합집합 감사·다음 후보](reports/policy_search_audit.json). 새 확인 전에는 채택하지 않으며 원래 정책/판정을 바꾸지 않았다. |
| SD와 예측구간 | [불확실성 진단](g1_quality/uncertainty_diagnostics.json), [Cal/Dev 구간 선택](g1_quality/interval_calibration_study.json) |
| 전체8학습·원본 전용 대조 | [최종 Dev 표](reports/neural_development_partial.json), [원본 전용 완료](reports/full_replay_control_complete.json) |
| 환경 정보만으로 충분한가 | [개봉 전 대조 설계](protocol/context_controls_frozen.json), [새 확인 대조](reports/fresh_context_controls.json) |
| 개봉 전 기준 | [동결 설계와 사전 수정 내역](protocol/study_design_frozen.json) |
| 실제 실행·의존성 호환 | [최종 두 이미지 CPU282검사](reports/final_code_validation.json), [standalone 당시275검사](reports/code_validation_current.json), [RGB48장 재생](verification/g1_quality/runtime_verification.json), [독립 묶음8장 재생](verification/standalone_g1/runtime_verification.json) |
| 파일 무결성 | [공개 파일 해시](evidence_inventory.json) |

파일 이름 `neural_development_partial.json`은 실행 중 이름을 보존한 것이며 현재 표에는8개 완료 실행이 있다. 일부 JSON의 개인 경로는 공개본에서 제거했고 SVG는 좌표·색을 유지하며 행끝 공백만 정규화했다. 원본 해시와 공개본 해시를 inventory에 따로 남겼다. 이후 추가한 이 README는 결과 파일 inventory의 대상이 아니다.

## 재현 범위

실험 자료·가중치는 로컬 연구팀 인계 묶음으로 보존한다. 코드·독립 upstream Git·인지 계보·품질/환경/보정과 RGB8예제만 마운트한 실행도 확인했다. 실제50GPU와40프로필 CPU의 비교이며 실제40GPU·종단 ROS 지연은 검증하지 않았다. 연구 후보가 실행되는 것과 채택 기준 통과는 다르다.

같은 도시·기존 링크의 다른 위치에서 확인했으며 완전히 새로운 도시나 모든 카메라 시야의 비중복을 주장하지 않는다. RGB/GT 경계 정합은 확인했지만 센서 노출 timestamp가 없어 시간 동기화를 보장하지 않는다. 온라인 ROI별 GT/IoU 계산·자동 학습 편입은 이번 결과가 아니다.
