# 기계학습 기반 인지 품질 추정을 통한 딥러닝 인지모델 개선 데이터 플라이휠 구축

인하대 스마트모빌리티공학과 · 2026학년도 2학기

기계학습·캡스톤디자인은 개발 결과를 공유하는 통합 프로젝트다.
기계학습 팀원은 안승현·손재호·장원태, 캡스톤 팀원은 안승현·장원태다.
이 저장소는 공통 코드·개발 환경·설계·입출력 규약·베이스라인을 관리한다.

## 목표와 현재 상태

**11월29일09시까지 [GP 대비·새벽 QA·shadow 작업](docs/experiments/대비특징_검수와_shadow_20261129.md)을 진행한다.** RF/GP를 둘 다 유지하며 최종 선택은 대용량 선별·재학습 비교 후로 미룬다. 아래 r2는 현재 검증 기준이다. [배포 모델](models/README.md)은 pull 후 사용할 수 있고 이후 실험 후보와 분리한다.

**11월28일 누적 갱신·새 확인을 완료하고 `m-v6-r2-20261128`을 개발 인지 기준으로 승격했다.**9영역·GP52 전체 IoU 추정·RF74 지원 오류 검수·e-means 맥락의 오프라인 쌍이 채택 기준과 독립 실행 묶음 재생을 통과했다. [결과·한계](docs/experiments/표시가용성_누적갱신_20261128.md) · [표·시각화·해시](docs/baselines/scene_cycle_20261128/confirmation/README.md) · [버전 등록부](config/perception_releases.json)

| 새 확인590장 | 결과 |
|---|---|
| 표시3구간 인지 IoU | 원래v6 **0.58353** → 부모r1 **0.76453** → r2 **0.78415** |
|10% 검수59장 | 근거리 오류107장 중 **53장(49.53%)**, 무작위 평균9.91%, Q0.7 대조40.19% |
| IoU 예측 MAE | 동결 GP52 **0.03136**, RF52 대조 **0.02421**. RF의 전이 우위를 함께 보고한다. |
| GP 불확실성 | 보정 구간·로그에 보존. SD 가산/정보 검수의 이득이 없어 기본 검수에는 넣지 않는다. |
| e-means / BO | e-means는 환경 맥락·다양성 로그. 환경별 임계 강제와 BO 기본 사용은 채택하지 않는다. |

96조건에서1,674프레임/4채널을 수집한 뒤3m thinning으로590장을 고정했다. 같은 지도 내 품질 fit과 최소46.25m 이격했지만 원래 인지학습 위치와는 최소0.10m이므로 완전한 새 도로/도시 일반화를 주장하지 않는다.05시 오류 회수는18/50,07시는0/4로 남은 약점이다. 자세한 분모와 원본 보고서를 보존했다.

**26R1 ROS 전방 수신과 초기 상주 추론을 연결했으며 다음은 장시간/주행 shadow 검증과 영어 최종 제출이다.** [간단 실행 명령](docs/guides/shadow_실행.md). RGB590장 선별·GT 사후 연결, 원래 작업공간 없이48장 release 재생을40프로필 CPU/5070 Ti GPU에서 확인했고 후속 shadow 소스를 포함한 두 의존성 이미지의 CPU345검사를 통과했다. 실제40계열 GPU·장시간/주행 shadow·자동 학습 편입은 후속이다. [작업·역할](docs/guides/일정과_역할.md) · [온라인 인계](docs/guides/온라인_연결과_최종제출.md)

기계학습·캡스톤 통합 프로젝트는 전방 RGB와 인지 출력에서 품질을 추정하고, 유용한 취약 데이터를 선별해 기존 성능을 보존하며 인지를 개선하는 데이터 플라이휠이다. **12월4일** 영상·수치·영어 발표·공개 페이지를 내부 마감한다. 팀원의 시작점은 [문서 목차](docs/README.md)다.

직전 전체 원본8학습·확인618장의 보류 결과와 이전4,096장 replay는 [실험 이력](docs/experiments/README.md)에 별도 보존했다. 가중치·품질 정답·정책·확인 모집단을 이번 결과와 섞지 않는다.

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

이 대조용 추론·시각화 명령의 기본값은 원래 M v6다. 현행 개발 인지는 [models](models/README.md)의 검증된 배포 모델을 기본으로 해석한다. 재학습/개인 전체 ledger를 사용할 때에는 `ML_ACTIVE_PERCEPTION_RUN` 또는 명시적인 `--adaptation-run`과 해당 버전 품질 번들을 함께 사용한다. [현재 버전 검사](src/quality_estimation/active_perception.py)는 등록된 가중치·계보를 확인한다. X/y 생성·회귀 학습·RGB 단독 품질 추론 CLI도 사용할 수 있다.

```bash
git submodule update --init
cp config/local.env.example config/local.env
# local.env에 자기 환경의 데이터·체크포인트·이미지를 설정한다.
scripts/run_container.sh python3 -m visualization.visualize_inference --split val --name m_v6_val
```

결과는 기본적으로 레포 상위 `outputs/inference_vis/<name>/`에 생성된다.
데이터·가중치·외부 제공 proto의 준비 조건과 수집 명령은 [개발 환경](docs/guides/개발환경.md)을 따른다.
이 코드는 다른 프로젝트 저장소를 import하거나 경유하지 않는다.

## 현재9영역 품질 파이프라인

버전별 X/y 생성, GP/RF 적합, RGB 단독 검수와 사후 GT 연결은 [데이터·특징·정답 가이드](docs/guides/데이터_등록과_특징정답.md)와 [표시 가용성·누적 갱신 재현](scripts/experiments/scene_cycle_20261128/README.md)을 따른다. [직전 주기 재현](scripts/experiments/environment_cycle_20261127/README.md)은 보존한다. 특징 순서는 각 번들 manifest가 정본이다. 현재 Q52/오류74열의 입력 합집합은74열이며 이전 f0/f1과 인지 신원·목표·정책을 섞지 않는다.

모델·RGB예제를 포함한 로컬 배포 묶음을 `ML_RELEASE_ROOT`로 지정하면 먼저 아래 검증을 실행할 수 있다. 연구 후보 묶음의 정상 실행은 채택 통과와 다르다.

```bash
python3 scripts/verify_environment_release.py --root "$ML_RELEASE_ROOT"
```

초기 `build_dataset` CLI와 X/y 설계는 [이력](docs/archive/X_y_데이터셋.md)으로 보존한다. 현재9영역 번들의 시작점으로 사용하지 않는다.

## 문서와 데이터 관리

- 공통 설계·평가·역할·개발 방법은 이 저장소의 `docs/`에 둔다. 미확정 설계와 측정된 결과를 구분한다.
- 강의 원문·개인 분석·과목별 발표 준비는 저장소 밖 로컬 워크스페이스에서 관리한다.
  팀 개발에 필요한 결론은 공통 문서에 반영해 로컬 자료 없이도 읽을 수 있게 한다.
- 원본 데이터·영상·마스크·feature·전체 학습 원장·실험 출력은 Git에 올리지 않는다. 사용자가11월28일 명시한 [검증된 배포 인지 가중치](models/README.md)는 버전·해시를 고정해 예외로 포함한다.
- 개인 경로와 컨테이너 이미지 이름은 Git에서 제외한 `config/local.env`에 기록한다.
- 코드 출처·revision과 공개 가능한 재현 근거는 [이관 기록](docs/archive/외부코드_선별.md)에 남긴다.
