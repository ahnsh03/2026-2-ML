# 9영역 특징 선정 근거

[실험 보고서](../../experiments/ROI_9영역_특징선정_20261124.md)와 [선정 특징·모델 해시](../../../config/region_feature_recipe_v1.json)를 함께 읽는다.

- `summary.json`에 데이터 역할·시각 범위·원본 수·CPU 비용을 기록했다.
- `quality_metrics.json`, `paired_validation.json`, `seed_summary.json`에 회귀·경보·검수·위치 재표집·RF 시드 반복이 있다.
- `development_comparisons.json`에 특징군·축소·관계·튜닝 비교를 남겼다.
- `environment_metrics.json`, `cluster_summary.json`에 RGB 분류와 e-means/K-means 관측이 있다.
- `context_distance_summary.json`은 고정 상부·보닛의 출력과 원·근거리 GT TP·FP·FN 진단이다.
- PNG 여섯 개는 실제 자료와 위 집계에서 생성했다. 원본 대응·상세 예측·가중치는 외부 실험 저장소에 보존한다.

새 자료는 고유 480장이다. 전체·학습 시각·미학습 시각 집계는 서로 겹친다. 기존 진단 자료는 앞선 연구에서도 열어본 기록이며 새 위치는 동일 시뮬레이터·카메라의 8위치다.
