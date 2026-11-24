# 기본 ROI와 특징 추출 실험

**11월 24일 후속 결정으로 기본 ROI를 `roi9-front-v1`로 정한다.** 상부를 y=169에서 높이로 반반 나눈 두 층·원거리·근거리 각 좌우와 보닛 하나다. 안승현·손재호는 이 마스크에서 특징을 공동 선정하고 장원태는 같은 특징표로 RF·GP와 군집·정책을 비교한다. 27일 첫 결합, 30일 오프라인 판정 일정은 유지한다. [비교 결과와 한계](../experiments/ROI_7영역_10영역_비교_20261124.md) · [일정과 역할](일정과_역할.md)

재호가 병행한 7영역 연구는 전체 화면의 상위 엔트로피·최대 구간 엔트로피·확률 마진을 더한 99열을 해당 실험의 기본값으로 선택했다. 이 결과와 7영역 전용 추출 경로는 [7영역 최종 기록](../experiments/ROI7_최종_피처_판정_20261124.md)과 [99열 명세](../baselines/roi7_final_20261124/selected_features.json)에 보존한다. 영역 공통인 세 불확실성 요약을 9영역에서도 대조했고 현재 27열에 포함했다. 두 연구의 모델·열 순서는 각각 보존한다.

## 고정한 영역과 비교 이력

| 구성 | 픽셀 배정과 용도 |
|---|---|
| **9영역 기본** | 상부 위·아래, 원거리·근거리 각 좌·우와 보닛. 가로선 y=169, 좌우선 x=640이다. [프로파일](../../config/roi9_front_v1/profile.json) |
| **7영역 이력** | 상부가 한 층인 이전 정의와 결과를 재현할 수 있다. [프로파일](../../config/roi7_front_v1/profile.json) |
| **전체 화면 통계 추가** | 보닛을 포함한 화면 전체에서도 같은 12개를 계산한다. 현재 시작 기준선은 `(9+1)×12=120`열이다. 전체 화면 통계는 별도의 집계 범위다. |
| **10영역 보존** | 세 띠를 좌·중앙·우로 나눈 마스크와 기존 실험은 재현용으로 유지한다. 기존 영역별 모델은 이 프로파일로 실행한다. [프로파일](../../config/roi10_front_v1/profile.json) |

고정 전방은 1280×720, 수평 화각 90°, 피치 2°이며 수평선은 y=338이다. 수평선과 보닛 사이를 1대2로 나눈다. 원·근거리는 영상의 띠를 뜻한다. 상부 위층은 0≤y<169, 아래층은 169≤y<338이며 세 프로파일의 원·근거리와 보닛 경계는 픽셀 단위로 같다. 공간 구획이므로 건물·하늘의 실제 의미론적 마스크를 보장하지 않는다. 7영역 경계는 10영역의 중앙 칸 내부를 지나므로 10영역 통계만으로 7영역을 정확히 복원할 수 없다.

현재 9영역 결정은 상부의 조명·색과 지평선 주변 구조물을 나누어 실험하자는 프로젝트 결정이다. 이전 7영역 선택은 과거 비교 근거로 보존한다. 일부 주행과 같은 특징을 사용한 GP에서는 10영역이 더 좋았다. 선택에 사용한 도로는 개발·진단 자료로 남기고 최종 미사용 도로를 별도로 확보한다. RF·GP의 최종 채택과 환경별 검수 기준은 후속 판정이다.

## 이전 7영역 연구의 99열 재현

동일 프레임의 전방 RGB, 4×180×320 uint8 인지 맵, 고정 `f0`의 4×384×640 float32 마킹 softmax를 사용한다. 입력의 위아래 패딩 12픽셀씩은 추출기가 제외한다. 출력 JSON의 열 순서를 [확정 목록](../baselines/roi7_final_20261124/selected_features.json)과 대조할 수 있다.

```bash
PYTHONPATH=src python3 -m quality_estimation.spatial_features \
  --features roi7-iou99 --roi-profile config/roi7_front_v1 --image front.png --head-maps heads.npy \
  --marking-probability marking_probability.npy --output roi7_iou99.json
```

## 지금 바로 추출할 공통 특징

| 특징군 | 열 이름과 정의 |
|---|---|
| RGB 8개 | `intensity` 회색조 평균, `chroma` RGB 최댓값−최솟값 평균, `intensity_std` 밝기 표준편차, `dark_channel` 다크 채널 중앙값, `local_contrast` 국소 대비 평균, `sharpness` Sobel 선명도, `glare_fraction` 245/255 이상 밝기 비율, `median_residual` 중앙값 필터 잔차 평균 |
| 출력 4개 | `drivable_fraction` 주행가능 면적, `lane_fraction` 차선 면적, `lane_probability` 흰·노란 확률 합의 **영역 전체 평균**, `marking_entropy` 마킹 엔트로피 평균 |
| 이번에 추가 비교한 RGB 5개 | `dark_fraction` 밝기 <40/255 비율, `intensity_p10`, `intensity_median`, `intensity_p90_minus_p10`, `saturation_median`. 공통 12개에 모두 추가한 `robust17` 후보로 비교했다. 최종 채택한 특징 목록으로 간주하지 않는다. |

[공통 공간 추출기](../../src/quality_estimation/spatial_features.py)는 7·9·10영역을 지원하며 기본 프로파일은 9영역이다. RGB를 320×180으로 면적 축소하고 고정 v6의 4×180×320 바이트 출력 맵에서 집계한다. GT·날씨·시간·위치를 입력받지 않는다. 실제 수치 정의는 [기존 RGB·맵 함수](../../src/quality_estimation/roi10.py)를 공유한다.

```bash
PYTHONPATH=src python3 -m quality_estimation.spatial_features \
  --image front.png --head-maps heads.npy \
  --include-global --output spatial_features.json
```

출력은 프로파일·마스크 해시·계산 해상도·순서 있는 `영역.특징` 목록과 값이다. 위 명령은 120열을 낸다. 이번 `robust17` 후보는 `--features robust17`을 지정하고 전체 화면 옵션을 생략한 153열이다. 과거 10영역 비교 재현에는 `--roi-profile config/roi10_front_v1`를 명시한다. 출력 경로는 새 파일이어야 한다. 최초 X56의 영역당 8개는 RGB 4개와 출력 4개였으며 물리 ROI의 12개와 다르다. [X56 원문](../archive/X_y_데이터셋.md)

## 피처 선정 당시의 후보와 담당

| 담당 | 비교할 후보 | 참고와 측정 목적 |
|---|---|---|
| **재호** | 이번 RGB 5개와 색상·채도·대비·선명도의 분포, 다크 채널 | Roser·Cheng·Lu의 환경 단서. 다른 도로에서도 환경을 안정적으로 구분하는지 본다. |
| **승현** | 예측 차선 확률 마진, 성분 크기·연결성분, 경계·내부 엔트로피, 필요 시 raw logit energy | MetaSeg·SEE의 품질 단서. 같은 환경 내 정상·저하를 구분하는지 본다. |
| **공동** | 보닛–노면 밝기차, 원–근거리 대비차, 좌우 비대칭과 영역 통합 | 남은 독립 특징에 관계를 더하고, 가로 분할을 합쳐도 선별 능력이 유지되는지 본다. |

논문별 원래 과제·정의와 재현 범위는 [논문 목록](../baselines/참고논문_베이스라인.md), [적용 분석 원문](../archive/수집데이터_재가공과_논문활용_20261123.md)에 있다. RGB 환경군 입력과 품질 회귀 입력은 각각 선정한다. 필터 잔차에는 재질·반사·압축 효과도 섞인다.

1. **9영역에서 후보를 일괄 계산한다.** 120열 기준선을 남기고 모든 ROI에 같은 후보를 계산한다. RGB만·출력만·결합 입력과 후보군 하나씩 추가를 같은 데이터·튜닝 예산의 RF·GP로 비교한다.
2. **영역별 특징을 줄인다.** 상부·보닛의 출력 4개 → 차선 면적 1개 → 0개를 비교하고 도로의 RGB·출력도 묶음별 제거한다. 남은 열만 연결하므로 차원이 줄어든다. RF 중요도와 다시 학습한 제거 실험을 함께 본다. 최종 추출기에서 불필요한 통계 계산도 제거한 뒤 비용을 다시 잰다.
3. **남은 특징에 관계를 추가한다.** 보닛–노면·원–근거리·좌우 차이를 하나씩 더한다. 직접 입력에서 빠진 열도 관계 계산 재료로 남길 수 있다. 관계의 효과와 영역 통합의 효과는 각각 대조한다.
4. **좌우를 통합하는 후보를 비교한다.** 9영역의 좌·우를 띠별로 합쳐 상부 위·아래·원거리·근거리·보닛 5영역으로 집계하거나 일부 띠만 합친다. 전체 화면 옵션은 동일하게 둔다. 파생 마스크·열 이름·버전을 기록하고 고정 9영역 기준선과 비교한다. 기존 10영역 좌·중앙·우 통합은 과거 결과 해석이 필요할 때 보조 대조로 둔다.
5. 개발 도로에서 최종 열을 고정하고 별도 보정 자료에서 GP 불확실성·환경군·검수 정책을 조정한다. 미사용 도로 GT 평가와 동일 예산 재학습까지 오프라인으로 마친 뒤 온라인에 연결한다.

영역을 합칠 때 원본 RGB·예측 맵과 파생 마스크로 다시 집계한다. 평균은 픽셀 수 가중, 분산은 모멘트 결합이 가능하지만 분위수·중앙값·선택 픽셀 통계는 기존 값의 평균으로 대체할 수 없다. 관계 추가·열 제거·영역 통합을 한 번에 바꾸면 원인을 구분하기 어려우므로 각각의 기준선 비교표를 남긴다.

한 프레임은 한 행이고 열은 `영역.특징`이다. 예측 픽셀 위 통계는 예측이 없을 때 대체값·면적·유효 여부를 함께 기록한다. 필터·경계는 전체 영상에서 계산한 뒤 집계해 ROI 테두리가 특징이 되는 것을 막는다. 회귀 정답과 최종 출력은 **전체 차선 IoU 하나**이며 ROI별 IoU를 온라인 목표로 추가하지 않는다.

## 11월 24일 특징 선정 인계

품질 회귀는 **RGB 10열+인지 출력 요약 14열+원본 확률 요약 3열의 공통 27열**을 RF·GP에 제공한다. 날씨 분류는 80개 히스토그램 성분, 지정 시각 분류는 160개 성분, e-means 문맥은 상부 위층 좌우·보닛 RGB 54열을 사용한다. [실험 결과와 선정 근거](../experiments/ROI_9영역_특징선정_20261124.md), [순서 있는 열·모델 해시](../../config/region_feature_recipe_v2.json)가 현재 인계 기준이다.

공통 120열은 후보 비교의 시작점이며, 기본 27열은 아래 별도 추출 경로로 실행한다. 후보 통계와 모델 입력은 이름·순서·프로파일 해시를 함께 대조한다. RGB 후보는 기존 8개+분위수·채도 5개+색차·채도 IQR·채도 가중 원형 hue 두 성분·경계 밀도 5개로 18개다. 출력 후보는 기존 4개와 차선 가중 엔트로피·확률·차선/주행가능 불일치·예측 지원 여부의 총 8개다.

```bash
# FROZEN_MODEL_ROOT는 공유받은 export/selected_v2/models 디렉토리다.
PYTHONPATH=src python3 -m quality_estimation.region_inference \
  --image front.png --head-maps heads.npy \
  --marking-probability marking_probability.npy \
  --recipe config/region_feature_recipe_v2.json --task quality_gp \
  --model "$FROZEN_MODEL_ROOT/quality_gp.joblib" --output quality.json

# 환경 특징과 태그는 RGB만으로 계산한다.
PYTHONPATH=src python3 -m quality_estimation.region_inference \
  --image front.png --recipe config/region_feature_recipe_v2.json \
  --task weather --model "$FROZEN_MODEL_ROOT/weather.joblib" --output weather.json
```

`--task quality_rf`, `hour`, `environment`도 지원한다. 환경 군집 아티팩트는 `environment.npz`다. 모델 파일을 생략하면 선정한 특징값을 JSON으로 저장한다. 입력 RGB는 원래 1280×720이며, 품질 입력은 고정 v6와 기존 전처리에서 만든 4×180×320 uint8 요약 맵과 같은 프레임의 4×384×640 float32 마킹 softmax다. 원본 확률은 추출기 안에서 패딩 12행씩을 제외한다. 원본 확률 통계는 엔트로피 상위 10% 평균·최대 32×32 구간 평균·예측 차선 확률 마진이다. 전체 화면 통계이므로 상부·보닛의 출력도 포함한다. CLI는 아티팩트·프로파일·입력 차원과 출력 파일 충돌을 검사하고 입력 해시를 기록한다. NPY의 생성 버전은 데이터셋의 검증 영수증과 연결한다. 24열 v1과 27열 v2의 모델·명세를 함께 묶어 사용한다. 추가 독립 도로에서 정상 장면 과소추정과 GP 구간의 과소포함이 남아 있어 27열은 27일 결합 기준선으로 인계한다.

후보 표를 새로 만들 때의 실행 경로는 다음과 같다.

```bash
PYTHONPATH=src python3 scripts/extract_region_candidates.py \
  --source-dir "$RECORDED_SOURCE" --reference-packs "$VERIFIED_REFERENCE_PACKS" \
  --media-root "$RGB_ROOT" --output-dir "$CANDIDATE_PACKS" \
  --uncertainty-packs "$RAW_UNCERTAINTY_PACKS"
```

| 입력 | 구조 |
|---|---|
| `RECORDED_SOURCE/protocols/<key>.jsonl` | `sample_id`, `image`, `expected_sha256.rgb`, `pixel_sha256`를 가진 RGB 목록 |
| `RECORDED_SOURCE/maps/<key>.npz` | 같은 순서의 `ids`, `maps`와 `checkpoint_sha` |
| `VERIFIED_REFERENCE_PACKS/<key>.npz` | 같은 순서의 `y`, `ids`, `pixel_sha256`, `group`, `site`, `cohort`, `weather`, `hour`, `xy` |
| `RAW_UNCERTAINTY_PACKS/<key>.npz` | 같은 `ids`, float32 `values` N×3, `ordered_features`, `checkpoint_sha`와 NPZ SHA-256 JSON 영수증 |
| reference JSON 영수증 | NPZ·manifest·출력 맵의 SHA-256 |

reference는 정답과 역할 메타데이터를 연결하는 자료다. 특징은 RGB와 인지 맵에서 9영역으로 다시 계산한다. 출력은 `rgb` 180열·`output` 80열·`relations` 88열·`histograms` 500열, 선택 옵션의 원본 확률 요약 3열과 별도 정답·메타데이터다. 원본 확률 요약은 같은 전처리·가중치·프레임에서 계산하고 값과 ID·열 순서·가중치·영수증 해시를 대조한다. 옵션을 생략하면 원래 848열 후보 캐시를 재현한다. 스키마와 원본·모델·팩의 해시를 보존한다.

비교는 `compare_region_candidates.py`, 내부 순위 기반 축소는 `reduce_region_candidates.py`와 `reduce_region_environments.py`, 축소 후 관계 비교는 `compare_compact_relations.py`를 사용한다. 환경 군집은 `compare_region_environments.py`, 고정 후보 평가는 `evaluate_region_candidates.py`다. 각 명령의 `--help`와 결과 문서의 데이터 역할·설정 예산을 함께 따른다. 이후 보정·검수 정책의 인계는 [환경 군집과 검수 정책](환경군집과_검수정책.md)에 연결한다. 실제 설정과 27열 재적합 경로는 [프로토콜 묶음](../baselines/roi9_features_20261124/protocols/README.md)에 있다.

## 인계와 완료 기준

인계물은 추출 코드, 순서 있는 열 목록, 같은 ID의 특징표, 데이터·인지모델·ROI 버전, RF·GP의 MAE와 검수 결과, 계산시간이다. 새 추출기의 7·9·10영역 경계·기존 함수 대조·전체 화면 통계·분위수·빈 예측을 검증했다. 직접 회귀 비교는 [RF·GP 재현 경로](품질모델_학습과_불확실성.md)로 실행할 수 있다. 배포용 번들과 온라인 연결은 30일 오프라인 판정 후 완성한다.

현재 27열 비교를 다시 적합할 때는 원본 확률 3열을 포함한 팩을 제공한다.

```bash
PYTHONPATH=src:scripts python3 scripts/compare_compact_relations.py \
  --packs "$CANDIDATE_PACKS" \
  --protocol docs/baselines/roi9_features_20261124/protocols/raw_uncertainty_protocol.json \
  --output-dir "$QUALITY_REFIT_RUN" --jobs 4
```
