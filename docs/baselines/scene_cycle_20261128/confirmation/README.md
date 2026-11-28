# 11월28일 누적 갱신 최종 확인·인계

**m-v6-r2-20261128 + GP52 Q + RF74 검수**가 사전 고정한 오프라인 기준과 RGB·standalone 실행 검증을 통과했다. e-means는 환경 맥락 로그이며 BO는 기본에서 제외했다. live ROS·자동 학습 편입은 후속이다. [전체 해석](../../../experiments/표시가용성_누적갱신_20261128.md) · [작업과 역할](../../../guides/일정과_역할.md)

## 먼저 읽을 근거

| 항목 | 파일 |
|---|---|
| 최종 채택/버전·모델 | [채택](reports/adoption.json) · [등록부](reports/perception_releases.json) · [Q/오류/환경 입력 순서](reports/active_review_manifest.json) |
| 인지 점수·유지 검사·대조 | [새 확인](reports/perception_confirmation.json) · [실제 GPU 유지 검사](reports/gpu_retention.json) · [CSV](tables/perception.csv) |
| 검수·Q·구간·분모 | [7개 고정 정책](reports/review_confirmation.json) · [CSV](tables/review.csv) · [05/07시 분리](reports/explicit_sdk_clock_audit.json) |
| 군집·BO | [새 환경 확인](reports/environment_confirmation.json) · [공식 e-means 레시피](reports/emeans_recipe.json) · [같은 예산 BO](reports/equal_budget_bo.json) |
| 자료·누수·수집 | [등록](reports/registration.json) · [역할9개·36쌍](reports/reserved_role_audit.json) · [정합](reports/alignment.json) · [96주행 복원](reports/simulator_restoration.json) |
| 실제 인계 | [RGB590장/GT연결](reports/actual_rgb_runtime.json) · [standalone48장](reports/standalone_runtime.json) · [CPU331검사×2](reports/code_validation.json) · [종료 감사](reports/closeout.json) |
| 사전 고정과 보완 | [최종96조건 설계](reports/final_study.json) · [원래72조건 설계](reports/initial_study.json) · [05시 추가 기록](reports/dawn_amendment.json) · [모델/정책 동결](reports/frozen_models_policy.json) |

집계 JSON의 본문은 `evidence`, 원본 SHA·별칭은 `publication_projection`에 있다. 개인 경로를 별칭으로 바꾼 게시본과 실행 원본의 SHA를 [게시 manifest](publication_manifest.json)에서 구분한다. 경로 별칭 파일을 실행 원본과 동일한 해시라고 주장하지 않는다. `source_snapshot`은 게시 시점의 공개 CLI 구현이며 이전 부모 적합의 실제 실행 소스는 [개발 단계 보존본](../development/README.md)에 별도로 있다.

## 핵심 결과와 해석

96조건1,674프레임에서 사전3m thinning590장을 확인했다. 표시3구간 macro IoU는 v6/r1/r2가0.58353/0.76453/0.78415다.59장 검수로 지원 근거리 오류107장 중53장을 회수했다. 같은 예산 무작위 평균은9.91%, Q0.7 대조는40.19%였다. GP52의 전체590장 MAE0.03136보다 RF52 대조0.02421이 작았으며 RF의 전이 우위를 숨기지 않는다. 고정GT≥50의441장 지표와 구간 폭/포함률도 함께 보고한다.

Dev로 선택한 주 번들을 새 확인에서 더 좋은 대조로 바꿔 끼우지 않았다. GP 정보 검수는 이득이 없어 기본에서 제외했다. e-means의 SDK 날씨/조명 의미표는0.90674/0.62156로, 직접 RGB 분류0.97565/0.98434와 다르다. 군집이 정확한 시각 분류나 환경별 임계 이득을 뜻하지 않는다.

05시 오류는18/50,07시는0/4다. 원본 `dawn_near_error_recall`은 과거07시 전용 이름이며05시 성적으로 읽지 않는다. 지원 오류는 픽셀 기준의 공학적 목표로, 헤드라이트 가시성 정상·안전의 정답이 아니다. 같은 지도에서 품질 fit과 최소46.25m 분리했지만 원래 인지학습 ego와 최소0.10m이므로 완전한 새 도로/도시 일반화를 주장하지 않는다. bootstrap 범위는 이 고정 구간에 조건부이며 재학습 시드는 하나다.

동결 bundle의 연구 메타데이터 `adopted=False`는 보존한다. 실제 채택은 별도 채택/로컬 release 영수증에 기록했으며 이 플래그를 바꿔 자동 학습 편입을 켜지 않았다. 실제40계열 GPU·실시간 ROS·지연/드롭 검사는 완료되지 않았다.


## GP의 추가 활용 후보 — 사후 진단

근·원거리 동시 오류99장과 근거리 유지·원거리 오류342장을 나눠 보면 Q GP의 σ AUC는0.46813, GP 오류 가치 σ는0.63518이다. 큰 σ를 전반적 성능 저하의 기준으로 정하지 않는다. GP 근−원 오류 평균 차이의 AUC는0.87790, RF 평균 차이는0.87731이었다.

이 진단은 별도 동결한 **GP-only 오류GP74**를 사용한다. 현재 주 번들의 RF74 오류 head에는 GP 잠재 SD가 없고, 품질Q GP의 SD와도 다르다. 오류GP74와 주 RF74의74개 순서 있는 입력은 실제 manifest에서 같음을 확인했다. 같은 GP74 평균 대비와 자기 SD 정규화, RF/GP 패밀리 대조를 구분한다.

GP가 이미 출력한 차이 평균 `μd=μnear−μfar`와 잠재 분산 `σd²=Σnn+Σff−2Σnf`를 사용한 `z=μd/σd`는 AUC0.92808이었다. **근/원 오류 점수의 관계와 불확실성을 함께 쓰는 후속 후보**다. 단순 σ 확대·GP 정보 슬롯의 현재 무효 결과와 구분한다. 정규화 점수는 보정된 실패 확률이 아니며 두 조건부 오류 집단 밖의 분류 성능을 뜻하지 않는다.

[추가 진단 원본](reports/uncertainty_semantics.json) · [두 의존성 이미지의 동일 통계 재현](reports/uncertainty_dependency_reproduction.json) · [16번 그림](figures/16_gp_uncertainty_semantics.png). 이는 확인 개봉 후의 설명용 분석이며 새 selector/임계로 채택하지 않았다. 새 Dev/Cal의 조건부 보정과 새 독립 확인을 거친 뒤 실제 선별 이득을 판단한다. 현재 r2·GP52/RF74 정책·채택 결과는 그대로다.

## 최종 시뮬레이터 종료 확인

릴레이 종료의 마지막 Park 패킷이 native 조종 모드를 다시 UDP로 바꾸는 것을 발견했다. 모든 UDP 송신자를 멈춘 뒤 SDK Keyboard/Park를 마지막으로 복원하고 전용 수집기를 종료했다.96epoch의 작은 settling 오차가 누적된0.24349m도 최초 cohort 위치로 복원했다. 최종 nativeKeyboard/Park·정지·정상 창과 위치 오차0.00273m 미만을 확인했다. 수집/모델/검수 결과는 바꾸지 않았다.

[최종 종료 감사](reports/final_shutdown_verification.json) · [SDK 복원](reports/final_sdk_mode_restoration.json) · [최초 cohort 복원](reports/whole_cohort_restoration.json) · [실제 최종 native 관측](reports/native_state_after_sdk_fix.json). 팀 재현은 [종료 순서](../../../../scripts/experiments/scene_cycle_20261128/README.md#26r1-수집-종료-순서)를 따른다. 주 인계의 core/runtime/config/docker는 독립 재생한287b1b5와 같고, 이후 커밋은 실험 진단·종료 도구·문서 보완이다.

## 발표·웹페이지용 그림

| 그림 | 설명과 사용 범위 |
|---|---|
| [08 인지](figures/08_perception_fresh_confirmation.png) | v6/부모/후속의 같은 표시 구간 점수. 오른쪽 모든 헤드 변화와 원래 유지 검사는 다른 모집단이다. |
| [09 품질·구간](figures/09_quality_fresh_confirmation.png) | 고정GT 지원 모집단의 Q오차와 구간 포함률. |
| [10 검수](figures/10_review_fresh_confirmation.png) | 같은59/590 예산, 주 정책·RF·GP·정보·Q0.7·부모를 비교. 부모 정답 버전은 다르다. |
| [11 환경](figures/11_environment_fresh_confirmation.png) | 직접 RGB, e-means, Lloyd 날씨 혼동행렬. W0–W5는 수집 SDK preset 번호다. |
| [12 Q 산점도](figures/12_quality_scatter_fresh_confirmation.png) | 모든 정의된590장,05/07/13/20시를 표시. |
| [13 matched 재학습](figures/13_matched_retraining_fresh_confirmation.png) | 같은 부모·18k·환경 수의 Risk/Random 대조. 한 시드·조건부 결과다. |
| [14 근/원 오류](figures/14_neural_band_errors_fresh_confirmation.png) | FN·FP를 함께 제시. 프레임 사건 수와 pooled 픽셀 지표를 혼동하지 않는다. |
| [15 학습 로그](figures/15_training_traces.png) | 완료 실행의 원본 로그/해시를 검증. 손실 곡선 자체가 채택 지표는 아니다. |
| [같은 Dev 전후 사례](development_cases/05_near_errors.png) · [거리 맥락](development_cases/06_distance_dawn.png) | 부모와 고정한 동일4ID의 최종 r2 그림. [부모 그림](../development/figures/05_near_errors.png)과 비교한다. |
| [새05시 FP/FN](confirmation_cases/05_near_errors.png) · [근/원·조명](confirmation_cases/06_distance_dawn.png) | 확인 후 선택한 설명용 사례. GT 패널은 원본 white/yellow GT, 출력이 아니다. |

같은 이름의 SVG도 제공한다. 개별 사진은 `development_cases/paired_*`, `confirmation_cases/fresh_*`의 RGB/GT/예측/오류/GT mask이며640×360,16:9를 유지한다. 각 `cases.json`은 원본/팩/버전과 재추론 TP/FP/FN 검증 해시를 담는다. 전체 raw/GT 데이터셋·가중치는 포함하지 않으며 **선별한 실제 GT 시각화 래스터는 포함한다**.
