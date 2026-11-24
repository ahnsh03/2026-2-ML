# 기계학습 기반 인지 품질 추정을 통한 딥러닝 인지모델 개선 데이터 플라이휠 구축

인하대 스마트모빌리티공학과 · 2026학년도 2학기

기계학습·캡스톤디자인은 개발 결과를 공유하는 통합 프로젝트다.
기계학습 팀원은 안승현·손재호·장원태, 캡스톤 팀원은 안승현·장원태다.
이 저장소는 공통 코드·개발 환경·설계·입출력 규약·베이스라인을 관리한다.

## 목표와 현재 상태

전방 RGB와 TwinLiteNet+ Medium v6 출력으로 전체 차선 IoU를 추정하고 취약 장면을 검수·재학습하는 데이터 플라이휠을 개발한다. 팀원의 시작점은 [팀 문서 목차](docs/README.md), 내부 마감과 인계는 [일정과 역할](docs/guides/일정과_역할.md)이다. 11월 24일 ROI 결정, 27일 첫 결합, 30일 오프라인 판정 후 12월 4일에 영상·영어 발표·공개 사이트를 완성한다.

**현재 특징 실험은 9영역으로 진행한다.** 상부를 높이로 반반 나눈 두 층·원거리·근거리 각 좌우와 보닛을 사용한다. 전체 화면을 포함한 공통 120열에서 축소·관계·원본 확률 요약을 비교해 RF·GP 공통 27열을 선정했다. 환경 쪽은 날씨 80개·시각 160개 히스토그램 성분과 e-means RGB54다. [특징 선정과 독립 도로의 한계](docs/experiments/ROI_9영역_특징선정_20261124.md)를 첫 결합의 기준으로 사용한다. [현재 절차](docs/guides/ROI와_특징추출.md)와 [앞선 비교 이력](docs/experiments/ROI_7영역_10영역_비교_20261124.md)을 확인한다.

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
