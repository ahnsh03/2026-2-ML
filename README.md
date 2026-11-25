# 기계학습 기반 인지 품질 추정을 통한 딥러닝 인지모델 개선 데이터 플라이휠 구축

인하대 스마트모빌리티공학과 · 2026학년도 2학기

기계학습·캡스톤디자인은 개발 결과를 공유하는 통합 프로젝트다.
기계학습 팀원은 안승현·손재호·장원태, 캡스톤 팀원은 안승현·장원태다.
이 저장소는 공통 코드·개발 환경·설계·입출력 규약·베이스라인을 관리한다.

## 목표와 현재 상태

전방 RGB와 TwinLiteNet+ Medium v6 출력으로 전체 차선 IoU를 추정하고 취약 장면을 검수·재학습하는 데이터 플라이휠을 개발한다. 팀원의 시작점은 [팀 문서 목차](docs/README.md), 내부 마감과 인계는 [일정과 역할](docs/guides/일정과_역할.md)이다. 11월 24일 ROI 결정, 27일 첫 결합, 30일 오프라인 판정 후 12월 4일에 영상·영어 발표·공개 사이트를 완성한다.

**현재 결합 기준선은 9영역·28개 특징·GP다.** 상부를 높이로 반반 나눈 두 층·원거리·근거리 각 좌우와 보닛을 유지한다. 고정 v6에서 전체 IoU를 추정하고 근·원거리 recall을 보조 진단한다. 새26R1 품질 학습 미사용4링크96장의 MAE는 GP0.0645·RF0.0712였지만 다른 보관 도로에서는 RF가 좋았고 원거리 불확실성 과신이 남았다. [최신 결과와 한계](docs/experiments/ROI9_GP와_26R1_검증_20261126.md), [현재 추출·실행](docs/guides/ROI와_특징추출.md), [열·모델 명세v3](config/region_feature_recipe_v3.json)를 따른다.

5070Ti용 `rtx50`과40GPU용 `rtx40` 이미지를 분리했다. 현재 시뮬레이터는 **26R1**이며 UDP 크기·조향 부호를 버전별로 나눈다. [환경 준비](docs/guides/개발환경.md) · [통신 설정과 미검증 항목](config/morai/README.md). RGB 단독 점수→후보→사후GT 연결은 재생 검증했고, 최종 검수정책·인지 재학습 채택·실시간 ROS 통합은 후속이다.

환경 RGB54 e-means·날씨80/시각160과27열v2는 [24일 연구 이력](docs/experiments/ROI_9영역_특징선정_20261124.md)으로 보존한다. 새 품질28열과 옛 환경·모델 아티팩트를 같은 명세로 혼용하지 않는다.

검수·동일 예산 재학습과 조건부 갱신 후보의 관측 결과는 [실험 결과](docs/experiments/README.md), 현재 실행·추가 구현은 [팀 가이드](docs/README.md#작업별-가이드)에서 확인한다.

## 저장소 구성

```text
docs/        guides · experiments · archive · baselines · contracts
config/      환경변수 예시·데이터 정책·센서·BEV 설정
docker/      수집 컨테이너 정의
tests/       데이터·모델 신원·학습/평가 누수·경보·선별의 회귀 검증
ros/         데이터 수집·UDP 브리지·메시지 서브모듈
scripts/     컨테이너 실행·동기화·감사·선별·가공
src/         모델 로더·기하 처리·시각화·품질 특징/정답·회귀·선별
third_party/ TwinLiteNet+ 서브모듈
```

## 빠른 시작 — M v6 추론 결과 시각화

인지 모델 추론·시각화의 기본값은 M v6다. X/y 생성·회귀 학습·RGB 단독 품질 추론 CLI도 사용할 수 있다.

```bash
git submodule update --init
cp config/local.env.example config/local.env
# local.env에 자기 환경의 데이터·체크포인트·이미지를 설정한다.
scripts/run_container.sh python3 -m visualization.visualize_inference --split val --name m_v6_val
```

결과는 기본적으로 레포 상위 `outputs/inference_vis/<name>/`에 생성된다.
데이터·가중치·외부 제공 proto의 준비 조건과 수집 명령은 [개발 환경](docs/guides/개발환경.md)을 따른다.
이 코드는 다른 프로젝트 저장소를 import하거나 경유하지 않는다.

## X/y 데이터셋 생성

`ML_OUTPUT_ROOT`를 용량이 충분한 외부 출력 볼륨으로 설정하고 실행한다. 기존 출력 폴더는 덮어쓰지 않는다.

```bash
scripts/run_container.sh python3 -m quality_estimation.build_dataset \
  --output-dir /outputs/quality_front
scripts/run_container.sh python3 -m quality_estimation.verify_dataset \
  --dataset-dir /outputs/quality_front --report /outputs/reports/quality_front_validation.json
```

X에는 GT가 들어가지 않으며 y와 sample_id로 대응한다. 기존 인지 train/val 노출 역할을 보존하고
품질 회귀 split은 별도로 정해야 한다. 세부 정의·산출물은 [X/y 데이터셋](docs/archive/X_y_데이터셋.md)을 따른다.

## 문서와 데이터 관리

- 공통 설계·평가·역할·개발 방법은 이 저장소의 `docs/`에 둔다. 미확정 설계와 측정된 결과를 구분한다.
- 강의 원문·개인 분석·과목별 발표 준비는 저장소 밖 로컬 워크스페이스에서 관리한다.
  팀 개발에 필요한 결론은 공통 문서에 반영해 로컬 자료 없이도 읽을 수 있게 한다.
- 원본 데이터·영상·마스크·feature·체크포인트·실험 출력은 Git에 올리지 않는다.
- 개인 경로와 컨테이너 이미지 이름은 Git에서 제외한 `config/local.env`에 기록한다.
- 코드 출처·revision과 공개 가능한 재현 근거는 [이관 기록](docs/archive/외부코드_선별.md)에 남긴다.
