# 개발·동결 은행의 중간 근거

**독립 확인 전의 개발 결과**다. 인지 부모는 `m-v6-r1-20261128`이고, 새 Dev334장과 별도 Cal835장으로 방법을 고른 뒤4,794장 후보 풀에서 각2,048장 은행을 동결했다. 인지 재학습 효과는 아직 이 표의 결과가 아니다.

| 질문 | 현재 관측 | 근거 |
|---|---|---|
| 전체 Q 회귀 |74열 GP/RF road-macro MAE0.04245/0.05148 | [모든 후보·정책](reports/development_selection.json) |
| 근거리 오류 회수 |34장 검수로45장 중31장, 무작위100회 평균10.44% | [같은 후보·정책 표](reports/development_selection.json) |
| 실제 RGB 재생 | GT 없이334장을 처리해 같은34장 선택 순서 재현 | [실행·GT 연결](reports/rgb_review_reproduction.json) |
| 선별 은행의 GT 품질 | Risk/환경 일치 Random의 근거리 오류588/315장, 평균 가치0.5371/0.3938 | [선별 후 GT 감사](reports/bank_gt_audit.json) |
| GP 불확실성 |σ 혼합의 개발 이득 없음. 근/원 동시 오류 AUC 약0.45–0.46 | [σ·대비 진단](reports/gp_uncertainty_diagnostic.json) |
| 군집 |photo12·e-means20pass/K16, ARI0.7496. 맥락 로그 후보이며 환경별 임계 이득은 없음 | [5/10/20pass·Lloyd 대조](reports/clustering_refinement.json) |
| BO | 같은8쿼리/초기4점/3시드에서 채택 기준 미달 | [효용 비교](reports/bo_effectiveness.json) |
| 자료 역할 |8역할의 ID/픽셀28쌍 교집합0, 부모 인지학습과 보호 역할 교집합0 | [역할 감사](reports/role_audit_parent.json) |
| RGB/GT 정합 |Sunny13 이동37프레임·3구간에서 자기 GT 경계가 이웃 GT보다 모두 가까움 | [콘텐츠 감사](reports/rgb_gt_registration.json) |

오류0은 안전/정상 라벨이 아니다.07시에는 이 Dev에서 근거리 오류 양성이 없어 recall을 계산하지 않았다. Q와 지원 오류는 다른 목표다. Q-GP와 오류-RF의 조합을 단일 GP/RF의 우승으로 설명하지 않는다. 같은 과제에서 행·열·예산을 맞춘 대조를 보존했다.

## 그림과 개별 사진

1. [동일 예산 BO 곡선](figures/01_equal_budget_bo.png)
2. [BO 채택 기준과 실제 이득](figures/02_bo_adoption_gate.png)
3. [GPσ와 오류 유형](figures/03_uncertainty_failure_types.png)
4. [Q 회귀와 지원 오류 회수](figures/04_review_model_development.png)
5. [근거리 FN/FP의 RGB·실제 GT·예측·오류](figures/05_near_errors.png)
6. [근/원 차이와07시 사례](figures/06_distance_dawn.png)
7. [GT 없이 고른 은행의 사후 GT 감사](figures/07_frozen_bank_gt_audit.png)
8. [실제 Semantic GT와 RGB 경계](figures/08_rgb_gt_registration.png)
9. [직접 RGB 분류와 군집 해석의 혼동행렬](figures/09_environment_context.png)

발표 배치용 [개별 사진](case_images/cases.json)은 case별 폴더에 `rgb.png`, `gt.png`, `prediction.png`, `errors.png`, `gt_mask.png`로 둔다. 평가 격자640×360이며16:9를 유지한다. 원본 카메라는1280×720이다. GT 패널은 예측을 재사용한 그림이 아니라 실제 source target이며 GT 픽셀 수를 팩의 TP+FN과 대조했다. 사례는 개봉한 Dev에서 설명용으로 고른 것이므로 평균 성능 근거와 구분한다.

## 프로토콜·보존

[본 설계](reports/protocol_study.json), [동일 부모·예산 학습](reports/protocol_matched_training.json), [head 분리](reports/protocol_head_split.json), [군집 최적화 보강](reports/protocol_cluster_refinement.json), [품질 인계 기준](reports/protocol_quality_guard.json)을 기록한다.20pass 설정은 공식 자동 스케일링에 의해 batch 크기·초기 step도 달라지므로 반복 횟수만 통제한 인과 실험은 아니다.

[품질 비교 분모 보강](reports/protocol_quality_guard_v2.json)은 신경망 후보·새 확인 개봉 전에 기록했다. GT≥50의 고정 모집단에서 버전별 품질 오차·구간을 비교한다. 빈GT FP 제거로 Q가 미정의가 되는 프레임 때문에 비교 분모가 바뀌는 문제를 피하며, 빈GT FP 검수나 전체 정의된 Q 보고를 제외하지 않는다.

[선별 영수증](reports/rgb_bank_admissions.json)은 실제 은행 비율42.72%와907장 교집합을 공개한다. 이 비율은 검수 평가의10%와 다르다. 원본과 부모의29,073장 replay는 두 은행에서 모두 보존한다. Coverage/24,000update는 별도 진단이며 Risk/Random의18,000update 비교에 섞지 않는다.

[적합 파라미터·열·모델 해시·모든 BO/무작위 탐색 경로](reports/models_and_search.json)와 [공개 보존 목록](publication_manifest.json)에 원본 보고서 SHA·공개 투영본 SHA·그림 SHA를 둔다. 전체 원본 데이터·모델은 포함하지 않는다. 일부 GT 설명 그림은 포함한다.40프로파일CPU·50실기GPU 검사·새 독립 확인·최종 채택은 [진행 기록](../../../experiments/표시가용성_누적갱신_20261128.md)에서 확인한다.

[실제 실행한 과거 소스](source_snapshot/README.md)는 기록된 SHA를 검증하기 위한 원문 보존이다. 실행에는 현재 `scripts/experiments/scene_cycle_20261128`의 도구를 사용한다.
