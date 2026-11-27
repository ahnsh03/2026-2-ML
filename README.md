# 기계학습 기반 인지 품질 추정을 통한 딥러닝 인지모델 개선 데이터 플라이휠 구축

인하대 스마트모빌리티공학과 · 2026학년도 2학기

기계학습·캡스톤디자인은 개발 결과를 공유하는 통합 프로젝트다.
기계학습 팀원은 안승현·손재호·장원태, 캡스톤 팀원은 안승현·장원태다.
이 저장소는 공통 코드·개발 환경·설계·입출력 규약·베이스라인을 관리한다.

## 목표와 현재 상태

전방 RGB와 TwinLiteNet+ Medium v6 출력으로 전체 차선 IoU를 추정하고 취약 장면을 검수·재학습하는 데이터 플라이휠을 개발한다. 팀원의 시작점은 [팀 문서 목차](docs/README.md), 내부 마감과 인계는 [일정과 역할](docs/guides/일정과_역할.md)이다. 11월 24일 ROI 결정, 27일 첫 결합, 30일 오프라인 판정 후 12월 4일에 영상·영어 발표·공개 사이트를 완성한다.

**현재 진행 단위:** 원본26,001장 전체 재생, RGB400 환경 분류와 e-means16군집, 같은 v6에서 시작하는8회 재학습과 새3도로 확인을 진행 중이다. [비교 설계와 진행 기록](docs/experiments/환경검증_전체원본재생_20261128.md)을 먼저 확인한다. 아래 숫자는 직전 완료 단위의 결과이며 새 후보의 채택은 아직 확정하지 않았다.

직전 단위에서는 **9영역·품질28열 GP와 환경 RGB27열·e-means8군집을 통합했다.** 신규3,072장+기존replay4,096장으로 방법3개×2시드와 기존자료 대조2회, 총8회(각약68분) 학습을 마쳤다. 개발에서 선택한 인지 후보의 주 확인IoU는0.57891→0.64997, 기존lane 손실은0.00294였다. 품질/군집 선별의 재학습 이득은 무작위보다 크지 않았다.

**새 f1/g1의 통합 교체는 보류하고 기존 f0/g0를 유지한다.** 새GP MAE0.07979가 유지 기준을 넘었고 군집·σ혼합 검수도 공통확률보다 나빴다. 후보 검수는 공통Q0.7확률순+감사로 되돌렸으며 환경별 하향 임계·자동 학습 승격은 채택하지 않았다. [통합 결과·보류 사유·다음 단위](docs/experiments/군집_검수_동일예산_재학습_20261127.md) · [공개 표·그림·해시](docs/baselines/selection_cycle_20261127/README.md)

현재 시뮬레이터는 **26R1**이다.5070Ti용 `rtx50`과40GPU용 `rtx40`을 분리하고 RGB단독→검수→사후GT·40CPU/50GPU·portable RF를 검증했다. liveROS·f1→f2 누적 학습은 후속이며 g1의 새 버전 오차와 보정을 먼저 보완한다. [개발 환경](docs/guides/개발환경.md) · [통신 인수](config/morai/README.md) · [담당·일정](docs/guides/일정과_역할.md)

이전7/10영역·RGB54·환경80/160열과26일 저속28+31 보정은 [날짜별 실험 이력](docs/experiments/README.md)으로 보존한다. 새 품질·환경·인지 버전을 이전 아티팩트와 섞지 않는다.

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
