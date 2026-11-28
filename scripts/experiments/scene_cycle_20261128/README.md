# 표시 가용성·누적 인지 갱신 재현

[현재 상태와 판정](../../../docs/experiments/표시가용성_누적갱신_20261128.md), [번들 계약](../../../docs/contracts/support_aware_review_bundle_v1.md), [원래 v6/개발 버전](../../../config/perception_releases.json)을 함께 확인한다. 실험 루트·인지 실행원·데이터·GPU프로파일은 각자의 환경변수로 전달한다. 원본과 체크포인트를 Git에 복사하지 않는다.

## 자료와 신원

실험은 고정9영역이다. 근/원은 영상 띠이며 미터 단위 거리나 가시성 정상의 GT가 아니다. 전체 Q와 별도 지원 오류 지시값을 학습한다. Q/오류/환경을 같은 버전의 번들에 묶되 Q와 오류의 최선 RF/GP head를 구분한다. RF-only·GP-only 대조도 보존한다.

이미 개봉한 이전 확인 자료는 fit으로 전환하고 시험 지위를 폐기한다. 새 Dev/Cal/학습 후보/확인은 서로 다른 구간과 역할을 유지한다. Scout의 GT지원량으로 표시 구간을 사전 확인한 사실을 공개하며 scout 프레임 자체는 최종 역할에서 제외한다. 원래 인지학습 좌표의 근접성도 별도로 보고한다. 같은 지도 구간 분리를 새 도시 일반화로 부르지 않는다.

## 실행 순서와 입출력

| 단계 | 도구 | 주요 입력과 산출물 |
|---|---|---|
| 제한 주행 수집 | `collect_cohort.py` | 사전 동결 plan, relay·SDK 설정, UTC hard stop. 매 epoch 상태/pose/날씨/시각을 복원하며 실패/부분 자료도 보존한다. |
| 역할 등록 | `register_cohort.py` | 완료 표식·4채널 원본·fit 팩. 전체/3m thinned index, ID·픽셀·지리·파일/복원 감사. |
| 콘텐츠 정합 | `alignment_audit.py` | 전체 index의 Sunny13. 같은 epoch의 이웃 GT만 비교하고 원본GT/경계 그림을 저장한다. 노출시각 동기화 증명은 아니다. |
| X/y 생성 | `../roi9_gp_20261125/extract.py` | 명시 인지 버전+immutable index. 전체 후보 RGB/출력/관계와9영역 TP/FP/FN. 새로운 인지 가중치에는 새 팩을 만든다. |
| 환경 RGB | `../environment_cycle_20261127/environment_features.py` | RGB-only index.9영역+전체의400열 히스토그램. |
| 환경/군집 적합 | `fit_environment.py` | fit3118의 RGB/GT표시 맥락·SDK태그. 지리 교차적합 확률, RGB/PCA16·prob11·photo12, e-means/Lloyd/K/3시드. |
| 충분한 군집 최적화 | `refine_environment.py` | 별도 사전 기록 후 같은 fit에서10/20pass를 비교. 원래5pass/Lloyd 대조는 덮어쓰지 않는다. |
| Q/오류 적합 | `fit_review_models.py` | 동일2500행·같은 인지 신원·52/74열. 과제별 RF/GP와 같은 HPO 예산. BO 효용이 없으면 `--search-methods random`을 사용한다. |
| 개발 선택 | `evaluate_development.py` | 새 Dev/별도 Cal·명시 환경 fit. 같은10% 검수 예산, FN/FP·도로별 가치·GP SD·환경만 아는 대조. |
| 번들 고정 | `build_support_bundle.py` | 선택한 Q/오류 head·환경 키·Cal·정책. 새 폴더에 해시 고정, `adopted=False` 유지. |
| GT 없는 은행 선별 | `select_training_banks.py` | GT필드가 없는 feature-only NPZ/RGB 목록. 부모 노출 제외, Risk와 SDK날씨×조명×출처 수를 맞춘 Random 목록을 먼저 동결한다. |
| GT 연결·누적 학습 | `quality_estimation.adaptation_cache`, `quality_estimation.retrain_streaming` | 선별 후 원본GT를 연결하고 전체 부모 replay를 보존한다. `--parent-adaptation-run`과 동결 부모 신원이 일치해야 한다. |
| 인지 비교 | `compare_perception.py` | 같은 backend/manifest의 원래v6·부모·후속 원시/보간 후보. Dev·유지 검사만으로 선택한다. |
| 새 버전 재적합 | 위 X/y/회귀/Cal/정책 도구 | 선택 인지 가중치로 모두 다시 생성한다. 같은 열이라는 이유로 이전 품질모델을 재사용하지 않는다. |
| 독립 확인 | 수집→등록→평가 | 인지·Q/오류·환경·보정·정책을 고정한 뒤 새96조건(05/07/13/20시)을 수집한다. 확인 결과로 재튜닝하지 않는다. |
| 시각화 | `plot_search.py`, `plot_review_diagnostics.py`, `plot_training.py`, `export_cases.py` | 동일 예산 BO·Q/오류·GP 불확실성·검증한 학습 곡선·실제 GT/예측/FP/FN. 학습 손실은 채택 지표와 구분한다. 인지/팩 신원·RGB 픽셀·재추론 TP/FP/FN을 검사하고 각 사진은16:9를 유지한다. |

순서상 의존성이 있는 작업은 직렬로 실행한다. GPU 주행 수집과 장시간 학습의 동시 실행에서 UDP stale이 반복되어 이번 실행은 수집 후 학습으로 자원을 분리했다. CPU 재가공은 인지 학습과 병행할 수 있다.

## 버전별 RGB 재생과 사후 GT

```bash
PYTHONPATH=src:ros/src/data_collection/src python3 -m quality_estimation.infer_versioned_review \
  --bundle "$REVIEW_BUNDLE" --adaptation-run "$PERCEPTION_RUN" \
  --manifest "$RGB_ONLY_MANIFEST" --data-root "$DATA_ROOT" \
  --output "$NEW_REPLAY_OUTPUT" --device cpu --batch-size 4 --budget 34

PYTHONPATH=src:ros/src/data_collection/src python3 -m quality_estimation.join_spatial_gt \
  --queue-dir "$NEW_REPLAY_OUTPUT" --packs "$VERSIONED_PACKS" \
  --source-index "$IMMUTABLE_SOURCE_INDEX" --key development \
  --data-root "$DATA_ROOT" --output-dir "$NEW_GT_REVIEW_OUTPUT"
```

실제 환경에 맞는40/50 컨테이너 안에서 실행한다. Docker DataLoader 작업에는 충분한 `--shm-size`를 지정한다. GT 연결은 원본바이트·픽셀·ROI·인지 신원을 검사하며 확인/보정/개발 프레임은 현 주기 학습에 편입하지 않는다.

## 공정 비교와 해석

확인 보고서의 기존 `dawn_near_error_recall`은 SDK07시만 뜻한다.05시를 포함한 일반 새벽 회수율로 읽지 않는다. `audit_clock_labels.py --review-assessment ... --output ...`는 원본을 보존한 채 `support_by_hour`에서05/07/13/20시를 명시해 집계하고 기존07시 정의와 대조한다. 이는 표기 보완이며 모델·선택·채택 기준을 변경하지 않는다.

- Coverage2048/24,000update는 먼저 시작한 진단이다. Risk2048와 matched Random2048은 같은 최종 후보 풀·같은 부모·전체29,073 replay·각18,000update를 사용한다. Coverage와의 차이를 순수 선별 이득으로 읽지 않는다.
- 검수 성능 평가의 예산10%와 실제2,048장 학습 은행의 풀 대비 비율은 다르다. 각 비율·교집합·조건별 수·선별 영수증을 보고한다.
- BO는8쿼리·공통 초기4점·3시드에서 무작위와 비교했다. 채택 기준을 넘지 못한 방법을 후속 갱신에 강제하지 않는다. 새 버전의 무작위 탐색은 별도로 기록한다.
- 환경 분류/군집 해석, 검수 이득, 실제 재학습 이득은 각각 다른 결과다. e-means를 맥락 로그에 쓴다는 사실만으로 환경별 임계의 이득을 주장하지 않는다.
- GP는 Q 경험 구간과 오류 선형 대비/잠재 공분산을 제공한다. σ를 실패 확률이나 근/원 오류 유형의 정답으로 읽지 않는다. σ 혼합이 Dev에서0.01 이상 이득이 없으면 기본 선별에 넣지 않는다.
- 출력 모델·칼리브레이션·정책 파일은 불변 경로와 해시로 구분한다. 실제40프로파일 CPU와50실기 GPU, 모델 버전 혼합 거부, standalone·GT 재연결은 학습/정책의 채택과 별도로 검사한다.
- `freeze_confirmation.py --quality-guard`에는 사전 기록한 고정GT 모집단의 기준을 명시한다. 이번 단위는 `quality_handoff_guard_gt_supported_v2.json`을 사용한다. 전체 정의된 Q와 빈GT FP는 계속 별도 평가한다.
- 새 확인 전에 primary·RF-only·GP-only·GP 정보 검수·공통Q0.7·부모r1·Lloyd 환경 대조의7개 번들을 고정한다. 마지막 대조는 같은 표현/K/시드의 군집 의미를 검사하며 공통 정책으로 둔다. e-means 군집 ID별 보정값을 Lloyd의 무관한 ID에 옮기거나 검수 정책 대체 후보로 해석하지 않는다. `assess_environment_context.py`는 fit에서 고정한 군집 해석표로 RGB 직접 분류와 군집별 날씨·SDK시각·표시 맥락을 각각 보고한다. SDK 조명 구간은 기존 학습과 같은 day/twilight/night 순서를 사용한다.
- `plot_confirmation.py`는 이미 고정·평가한 인지/품질/검수/환경 보고서를 같은 freeze 해시로 연결해 PNG/SVG를 만든다. 원래v6·부모·후속의 새 구간 IoU, 같은 GT 모집단 Q/구간, 고정 예산 검수와 환경 혼동행렬을 분리한다. 그림 생성으로 모델이나 기준을 다시 고르지 않는다.
- 확인 보고서의 `support_by_hour/segment/weather_hour`에는 GT와 예측 지원량, 무출력 FN, 빈GT FP, 근거리 precision/recall 유지와 원거리 오류, 각 집합의 선별 수를 함께 남긴다. 07시 자료를 무조건 실패나 정상으로 분류하지 않으며 양성이 없는 recall은 미정의로 유지한다.
- 인지 확인에는 Dev에서 미리 선택한 후보뿐 아니라 같은18,000update의 Risk/Random 원시 endpoint도 사전 고정한다. 새 구간의 paired condition-block 차이로 데이터 선별 효과를 별도 보고하고, 이 대조를 보고 채택 모델을 사후 변경하지 않는다. `assess_perception_confirmation.py --source-index`는 새 등록 RGB의 ID·픽셀·순서·역할과 수집 전 모델 고정 시점을 함께 검사한다.
- `decide_offline_adoption.py`는 같은 freeze/등록 모집단의 인지·검수 보고서에서 사전 선언한 주 정책/부모 fallback만 판정한다. 인지 개선과 검수+품질 보정을 따로 판단하며, 더 좋은 진단용 대조를 사후 채택하지 않는다. 판정 JSON 생성은 버전 등록·standalone 재생·온라인 연결이나 자동 학습 편입을 대신하지 않는다.
- 인지 비교도 전체 IoU뿐 아니라 `image_band_errors(_by_hour)`의 근/원 FP/FN·빈GT FP·무출력 FN·지원 수·pooled precision/recall을 보고한다. 미지원 precision/recall을1로 채우지 않는다. 새 분류 집계는 사전 IoU 채택 기준을 바꾸지 않는 보충 분석이다.
