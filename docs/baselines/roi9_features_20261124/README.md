# 9영역 특징 선정 근거

현재는 [실험 보고서](../../experiments/ROI_9영역_특징선정_20261124.md)와 [27열 v2 명세](../../../config/region_feature_recipe_v2.json)를 사용한다. [24열 v1 명세](../../../config/region_feature_recipe_v1.json)와 초기 결과는 1차 비교를 재현한다.

| 자료 | 포함한 내용 |
|---|---|
| `followup_summary.json` | 현재 27열·모델 해시, 624장·2,496채널 파일의 데이터 역할, 실제 고정·추론·집계 시각, CPU 비용·재현 대조 |
| `quality_followup.json`, `paired_followup.json`, `seed_followup.json` | 24/27/120열의 회귀·경보·검수, 첫 8위치 재표집, 27열 RF 5시드 반복 |
| `uncertainty_development.json`, `coverage_controls.json` | 원본 확률 3개 추가·단독·공통 입력과 사후 면적 제거 비교. 면적 제거는 추가 도로를 열어본 뒤의 개발 진단 |
| `window_review.json` | 주행 64프레임 창과 동일 날씨·시각 3장 소그룹을 각각의 실제 예산으로 비교 |
| `summary.json`, `quality_metrics.json`, `paired_validation.json`, `seed_summary.json` | 1차 24열의 데이터·비용·회귀·검수·위치 재표집·RF 반복 |
| `development_comparisons.json` | 1차 특징군·축소·관계·튜닝 비교 |
| `environment_metrics.json`, `cluster_summary.json` | RGB 날씨·시각 분류와 e-means/K-means 비교 |
| `context_distance_summary.json` | 상부·보닛 출력과 원·근거리 GT TP·FP·FN의 사후 진단 |

[검증 전표](validation.json)에 30개 검증·내부 링크·입력 중복·원본 확률 재추론 대조와 다섯 CLI의 실제 재생 결과를 기록했다.

[실험 설정 묶음](protocols/README.md)에 같은 예산의 특징·환경·불확실성 비교 설정과 해시를 보존했다.

`feature_allocation_v2.png`, `uncertainty_followup.png`, `isolated_road_cases.png`는 현재 27열의 배치·전이·한계를 보여준다. 다른 PNG 여섯 개는 고정 ROI와 1차 24열·환경 비교다. 원본 대응·상세 예측·가중치는 외부 실험 저장소에서 인계한다.

새 원본은 고유 624장이다. 첫 8위치 480장, 추가 독립 3링크 144장이다. 첫 480장의 전체·학습 시각·미학습 시각 집계는 서로 겹친다. 특징 연구에서 이미 살펴본 자료는 개발·전이 진단으로 표시하며 최종 미사용 평가와 구별한다.
