> 정리 전 원문이다. 현행 담당·일정·실행 경로는 [팀 문서 시작점](../README.md)을 따른다. 원문 경로와 보존 해시는 [이동 대응표](README.md)에 있다.

# 전방 영상의 품질 특징 X와 IoU 정답 y

2026-11-16 구현. 최신 ML 발표의 8종 특징을 전체 영상과 3행×2열 ROI에 적용했다.
`src/quality_estimation/`은 이 저장소의 인지 모델·기하·Semantic palette 코드를 사용한다.
다른 프로젝트 코드나 ROS 실행 환경은 필요하지 않다. 이어지는 회귀 학습·GT 없는 품질 추론·선별 평가는
[회귀 학습·선별 평가](회귀학습_선별평가.md)에 구현과 실행법을 정리했다.

## 1. 고정한 계산 규칙

실행 스키마는 [quality_xy_v2.json](../contracts/quality_xy_v2.json), 버전은 `ml-quality-xy-2.0.0`이다.
한 프레임의 **X는 56차원 float32**, y는 **통합 Lane IoU 스칼라 하나**다.
순서는 `global`, `r1c1`, `r1c2`, `r2c1`, `r2c2`, `r3c1`, `r3c2`이며 각 영역에 아래 8개 통계를 적용한다.

| 열 접미사 | 계산 |
|---|---|
| `lane_confidence` | 4클래스 argmax가 흰색·노란색인 픽셀의 `P(white)+P(yellow)` 평균. 예측 차선이 없으면 0 |
| `marking_entropy` | 영역의 모든 픽셀에서 4클래스 엔트로피를 `ln(4)`로 나눈 값의 평균 |
| `lane_area_fraction` | argmax 흰색·노란색 픽셀 수 / 영역 픽셀 수 |
| `lane_components` | 영역을 자른 후 8방향으로 연결된 차선 영역 수. 크기 필터 없음 |
| `brightness_mean` | 원본 RGB를 OpenCV uint8 grayscale로 변환한 평균 / 255 |
| `dark_fraction` | 원본 grayscale이 **40 미만**인 픽셀 비율 |
| `brightness_std` | 원본 grayscale의 모집단 표준편차 / 255 |
| `edge_density` | 원본 전체 grayscale에서 Canny(100, 200, apertureSize=3, L2gradient=True) 후 해당 영역의 경계 픽셀 비율 |

출력 특징은 384×640 letterbox에서 padding을 제거한 영상 영역에서 계산한다.
RGB 특징은 축소하지 않은 원본에서 계산한다. 두 공간에 같은 정규화 3행×2열 격자를 적용하되,
각 공간에서 `floor(k * size / n)` 경계로 나누고 끝 좌표를 포함하지 않는다. 빈 ROI는 허용하지 않는다.
표준화하지 않은 원시 특징을 저장하며, 이후 scaler는 품질 회귀 학습 그룹에서만 fit한다.
차선 없는 영역의 confidence=0은 낮은 확신도와 다를 수 있으므로 면적·연결 영역 수와 함께 해석한다.

모델은 [Medium v6 EMA epoch 11](../baselines/medium_v6_baseline_evidence.json)로 고정한다.
체크포인트 SHA-256 `a26311b3af4e92880f632509eff3c81defc1b95135f5b463c7a08e5b6e23d24b`를 확인하고 FP32로 추론한다.
RGB 입력은 uint8 BGR/BGRA PNG를 읽어 alpha를 버리고 기존 인지 학습과 같은 letterbox를 적용한다.
확률은 float32 softmax에서 직접 계산하며 시각화 PNG를 사용하지 않는다.

`y = TP / (TP + FP + FN)`은 기하·annotation valid 교집합에서 계산한다.
4클래스 argmax와 GT의 흰색·노란색 합집합을 비교하며 정지선 위치의 차선 오검출은 FP다.
union=0이면 `label_defined=false`, CSV 빈 값, NPY NaN으로 남긴다. 유효한 0점과 구별한다.

## 2. 구현 인터페이스

```python
from quality_estimation import extract_features, build_label

# GT 없는 X: original HxWx3 uint8 RGB, 4x384x640 float32 probabilities
x = extract_features(rgb, probabilities, geometry)  # shape (56,)

# y에만 GT와 annotation validity 사용: HxW uint8 class IDs, boolean masks
label = build_label(prediction, target, geometric_valid, annotation_valid)
```

`features.py`에는 GT·날씨·시각·위치·run ID 입력이 없다. `labels.py`는 정답과 TP/FP/FN을 별도로 만든다.
dataset builder는 정답 검증까지 함께 하므로 GT가 필요하지만, 운용용 feature 함수는 GT 없이 호출할 수 있다.

## 3. 생성·검증 명령

[개발 환경](개발환경.md)에 따라 `config/local.env`에서 데이터·체크포인트·컨테이너 이미지를 준비한다.
`ML_OUTPUT_ROOT`는 용량이 충분한 **원본 데이터와 별도인 출력 볼륨**으로 지정한다.
컨테이너 안의 `/data`, `/ckpt`, `/ml`은 읽기 전용, `/outputs`만 출력 루트다.

```bash
# 먼저 run별 2프레임으로 입력을 확인한다.
scripts/run_container.sh python3 -m quality_estimation.build_dataset \
  --output-dir /outputs/quality_smoke --max-per-run 2

# 기본값: v6 데이터셋 train+val 중 front만. 이미 존재하는 출력 폴더는 거부한다.
scripts/run_container.sh python3 -m quality_estimation.build_dataset \
  --output-dir /outputs/quality_front --batch-size 8 --num-workers 4

scripts/run_container.sh python3 -m quality_estimation.verify_dataset \
  --dataset-dir /outputs/quality_front --data-root /data \
  --report /outputs/reports/quality_front_validation.json

scripts/run_container.sh python3 -m unittest discover -s tests -v
```

`--split`은 기존 인지 데이터셋의 읽을 split을 고르는 옵션이다. 새 품질 회귀 split을 만들지 않는다.
`--group-map`에는 선택적으로 `{capture_run_id: parent_session_id}` JSON을 전달할 수 있다.
미지정 시 capture run을 그룹으로 남기며, 같은 주행에서 나눈 run이나 환경 변형은 학습 전에 상위 세션으로 묶어야 한다.
`--audit-per-run` 기본 1로 run별 확률·예측·GT fixture를 저장한다. 검증기는 최소 1개 fixture가 있어야 통과한다.

## 4. 출력과 재현 기록

| 파일 | 의미 |
|---|---|
| `X.npy`, `y.npy`, `label_defined.npy` | 순서가 같은 특징 행렬, 정답 벡터, 정답 유효성 |
| `features.csv`, `labels.csv` | sample_id별 특징과 정답·TP/FP/FN; 서로 분리 |
| `sample_ids.json`, `feature_names.json`, `schema.json` | 행/열 순서와 계산 규약 |
| `samples.jsonl` | RGB/Semantic/마스크 경로·SHA, capture·기하·정책·기존 노출 역할 |
| `audit/*.npz` | run별 첫 프레임의 float32 확률·예측·GT·valid |
| `dataset.json`, `summary.json` | 모델·소스 코드·입력 manifest·스키마·출력 해시, 버전·실행 환경·건수 |
| `run.json`, `_SUCCESS` | 완료 여부와 dataset metadata 해시 |

`_SUCCESS`가 없는 폴더는 실패 또는 미완료이므로 학습 입력으로 쓰지 않는다.
재실행할 때 새 출력 이름을 사용하고 검증 보고서는 완료 데이터셋 밖에 저장한다.
모든 프레임에서 원본 Semantic의 palette와 크기, RGB와 capture 대응, cached GT와의 nearest-resize 일치를 검사한다.
독립 검증기는 전체 CSV/NPY의 행 대응·수치·IoU 계산식과 manifest 해시를 확인하고,
run별 fixture에서 원본을 재독해 IoU 픽셀 수·특징 수식·padding 무관성을 다시 확인한다.

```python
import numpy as np
from pathlib import Path

root = Path("/outputs/quality_front")
X = np.load(root / "X.npy", allow_pickle=False)
y = np.load(root / "y.npy", allow_pickle=False)
defined = np.load(root / "label_defined.npy", allow_pickle=False)
# 학습 전에 samples.jsonl의 부모 세션·노출 역할을 검토하고 품질 split을 별도로 고정한다.
# undefined 행은 보존하되 지도 회귀의 정답으로 쓰지 않는다.
```

기존 train은 `f0_train`, 기존 validation은 `historical_dev`로 기록한다.
인지 모델 학습·선택에 이미 노출된 자료이므로 이를 품질 회귀의 독립 test로 이름만 바꾸지 않는다.
동일 RGB 해시 중복도 검증 보고서에 집계한다. 연속 프레임의 유사성까지 제거했다는 의미는 아니다.

## 5. 확인한 범위

원본 데이터 adapter·artifact·회귀·추론·분할·선별·재학습 역할·최종 잠금·모델 namespace 소유권까지 60개 검사가 통과했다. padding·홀수 ROI 경계·원본 RGB 통계·8방향 연결·빈 차선·정지선 FP·
유효한 IoU=0·GT 유효성과 특징의 분리·BGR/BGRA 호환을 확인했다.
18개 run의 36프레임 사전 실행에서 X=(36,56), y=(36,), 정답 36개가 생성됐고 독립 검증을 통과했다.

전체 실행은 **18개 run, 전방 9,135프레임**에서 완료했다. **X=(9135,56), y=(9135,)**이며
인지 학습 노출 8,667프레임과 과거 checkpoint 선택용 468프레임의 역할을 보존했다.

- 9,135프레임 모두 원본 Semantic과 cached GT 일치. 정답 undefined 0, 비정상 특징 값 0.
- 전체 CSV/NPY의 sample_id·행 순서·수치 일치, TP/FP/FN·IoU 계산식 및 입력 manifest/출력 해시 검증 통과.
- 18개 run의 fixture를 재독해 독립 IoU 카운트·전체 56종 특징 수식·padding 무관성 확인. 특징 최대 절대 오차 1.26×10⁻⁷ 미만.
- 동일 RGB 파일 해시 중복 0그룹. 후속 분할·선별은 decoded RGB 픽셀 해시와 시간·이동 간격을 사용하며, 유사한 장소가 모두 제거된다는 뜻은 아니다.
- IoU=0인 29프레임과 예측 차선이 빈 7프레임도 삭제하지 않고 보존. 전체 56개 열 중 상수 열 0.
- FP32 batch 8·worker 4의 전체 생성 경과 시간 약 480초. 입력 준비·I/O를 포함하며 실시간 추론 속도가 아니다.

이 9,135프레임 검증은 기존 인지 학습/개발 노출 자료의 X/y 처리 검증이다. 별도로 인지 모델 학습 이후 수집된
전방 16,044프레임에서 native Semantic 기반 X/y를 생성했다. 원본 재독 36프레임의 전체 56개 특징을
독립 수식으로 재검증했고 최대 절대 오차는 1.00×10⁻⁷ 미만이었다. 이후 수집 자료도 같은 adapter로 처리한다.
검증 조건은 `assert` 대신 명시적 예외를 사용하므로 `python -O`에서도 사라지지 않는다.
회귀 모델의 MAE와 예산별 Recall은 별도 개발 평가이며, 이 X/y 검증만으로 인지 모델 개선을 주장하지 않는다.

## 11/16 오후 버전 쌍과 다음 주기 연결

갱신 모델의 X/y 재생성·g1 학습·RGB 단독 추론은 검증된 `--adaptation-run`을 명시한다.
기본 f0는 유지하고, 서로 다른 인지 checkpoint와 품질 bundle의 연결은 거부한다.
개발 자료 전용 경보 보정, GT 없는 우선·무작위 감사 검수 대기열, 기존 성능 유지에 따른 갱신 판단은
[버전 쌍·경보·채택 판정](버전쌍_경보_갱신판정.md)을 따른다.

## 현재 파이프라인과 검증 이력

위 9,135행·16,044행은 초기 처리 검증 이력이다. 후속 신규 자료의 역할별 X/y 생성, 회귀 선택·예약 평가,
f1용 X/y와 g1 학습·경보·갱신 판단도 완료했다. [현재 개발 현황](개발현황.md)과
[통합 주기 검증](../experiments/통합주기_검증결과_20261116.md)이 현재 완료 범위다.
2026-11-22 코드 회귀 검사에서 전체 90개 테스트를 다시 통과했으며, 새 수집·학습 실험은 추가하지 않았다.
