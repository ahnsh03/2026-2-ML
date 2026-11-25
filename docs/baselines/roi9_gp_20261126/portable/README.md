# 40·50 환경의 RF 추론 호환 검증

새 RF의 joblib은 NumPy2.5.5·scikit-learn1.9.1에서 학습됐다. 40 이미지의 NumPy1.24.4·scikit-learn1.3.2에서 직접 로딩되지 않아, **학습된 트리의 분기·리프값을 그대로 NPZ로 내보냈다.** 기존 40 프로파일과 주 품질 GP는 유지했다. 내보내기는 원래 학습 패키지에서 실행하고, 실행 시에는 sklearn·pickle을 로딩하지 않는다.

| 검사 | 입력 | 결과 |
|---|---|---|
| 실제 품질 RF | 26R1 후속 평가96장·28열 | 원본 sklearn 대비 최대 예측 차이0 |
| 날씨·시각 RF | 26R1 RGB432장·각80/160열 | 클래스 변경0·최대 확률 차이0 |
| 고정 e-means 중심 | 같은432장·RGB54열 | 군집 변경0·거리 차이0, 중심 파일은 원본 그대로 |
| 40 이미지 실제 입력 경로 | RGB+저장된 f0 인지 출력 | 품질28열 재계산 최대 차이9.4e−10·예측 차이0, 날씨는 RGB만으로 통과 |

[CPU 학습 환경](validation_cpu312.json), [40 CPU](validation_rtx40_cpu.json), [50 CPU](validation_rtx50_cpu.json)에서 비교했다. 실제 40 GPU의 CUDA 실행을 확인한 것은 아니다. [범위·소스 해시](scope.json)

[호환 명세](../../../../config/region_portable_rf_recipe_v1.json)는 [v3 품질 RF](../../../../config/region_feature_recipe_v3.json)와 [v2 환경 모델](../../../../config/region_feature_recipe_v2.json)의 내보내기다. 이 명세는 주 GP 번들을 교체하지 않는다. 큰 NPZ와 대조 입력은 실험 볼륨으로 공유한다.

```bash
# 원본 학습 패키지 환경에서 한 번 내보낸다. 출력 폴더는 새 경로여야 한다.
PYTHONPATH=src python3 scripts/export_portable_forests.py \
  --quality-recipe config/region_feature_recipe_v3.json \
  --quality-model-root "$QUALITY_BUNDLE" \
  --environment-recipe config/region_feature_recipe_v2.json \
  --environment-model-root "$ENVIRONMENT_MODELS" \
  --output "$PORTABLE_BUNDLE"

# 40·50 환경에서 사용할 수 있는 RGB 전용 환경 추론 예시.
PYTHONPATH=src python3 -m quality_estimation.region_inference \
  --image "$FRONT_RGB" --recipe "$PORTABLE_BUNDLE/feature_recipe.json" \
  --task weather --model "$PORTABLE_BUNDLE/weather.npz" --output "$NEW_RESULT"
```

품질 RF에는 f0에서 생성한 `--head-maps`, `--marking-probability`와 실제 `--perception-checkpoint-sha256`도 필요하다. 아티팩트 해시·ROI·열 순서·인지 버전이 다르거나 입력에 NaN/Inf가 있으면 거부한다. 이 호환 포맷은 단일 출력 sklearn RandomForestRegressor/Classifier의 유한 입력 추론만 지원한다. 재학습·환경 임계값 채택·온라인 연결의 근거로 사용하지 않는다.
