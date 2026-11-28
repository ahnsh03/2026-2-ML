# 기계학습 기반 인지 품질 추정을 통한 딥러닝 인지모델 개선 데이터 플라이휠 구축

인하대 스마트모빌리티공학과 · 2026학년도 2학기

기계학습·캡스톤디자인은 개발 결과를 공유하는 통합 프로젝트다.
기계학습 팀원은 안승현·손재호·장원태, 캡스톤 팀원은 안승현·장원태다.
이 저장소는 공통 코드·개발 환경·설계·입출력 규약·베이스라인을 관리한다.

## 목표와 현재 상태

**11월28일19시까지 새 [표시 가용성·누적 갱신 주기](docs/experiments/표시가용성_누적갱신_20261128.md)를 진행 중이다.** 이전 유지성 통과 후보 중 최고인 `m-v6-r1-20261128`을 인지 개발 기준으로 승격했다. 이전 검수 정책은 보류하며, 새 Dev/Cal과 전체 부모29,073장 replay를 사용해 RF/GP·지원 오류·군집·선별·인지 갱신을 새 확인까지 검증한다. [버전 등록부](config/perception_releases.json)는 원래v6와 파생 버전을 분리한다. 아래 수치는 직전 완료 주기다.

전방 RGB와 TwinLiteNet+ Medium v6 출력으로 전체 차선 IoU를 추정하고 취약 장면을 검수·재학습하는 데이터 플라이휠을 개발한다. 팀원의 시작점은 [팀 문서 목차](docs/README.md), 내부 마감과 인계는 [일정과 역할](docs/guides/일정과_역할.md)이다. 11월 24일 ROI 결정, 27일 첫 결합, 30일 오프라인 판정 후 12월 4일에 영상·영어 발표·공개 사이트를 완성한다.

**11월28일 오프라인 검증 완료·통합 교체 보류:** 원본26,001장을 모두 보존하고 신규3,072장 은행4개를 비교해8회 재학습했다. 현재 후보는9영역·GP52열·공통0.7 검수와 정확한 동점의 표준화 margin이다. 독립 확인에서 인지·품질 유지 기준은 통과했지만 검수 정책은 실패해 온라인으로 넘기지 않는다. [상세 결과와 보류 사유](docs/experiments/환경검증_전체원본재생_20261128.md) · [표·16개 그림·해시](docs/baselines/environment_cycle_20261128/README.md)

확인618장 중 두 구간은 근거리 표시 GT가 없었다.62장 검수에서59장이 근거리 진단 미지원 장면에 몰렸고 실제 근거리 결함114장은 회수하지 못했다. 날씨 분류91.21%와 달리 시각 기반 조명 분류는69.86%였다. GP는 Dev에서RF보다 좋았지만 새 확인 MAE는 GP0.03653/RF0.03047로 역전됐다. 낮은 IoU 예측, 유용한 데이터 선별, 재학습 이득을 따로 판단한다.

**다음은 장면·타깃 가용성과 검수 정답 보완→새 독립 확인→조건 통과 후26R1 ROS shadow 연결**이다. 기본 v6 파일과 이전 버전 쌍은 덮어쓰지 않았다.40/50 프로필을 유지하고 두 의존성 이미지의 CPU312검사를 통과했다. 누적 부모·전체 replay·중복/계보 검사를 구현했으며 실제50GPU 역전파 smoke와 장시간 학습을 진행한다.40계열 실제 GPU와 실시간 지연·ROS 연결은 후속이다. [개발 환경](docs/guides/개발환경.md) · [역할·일정](docs/guides/일정과_역할.md)

직전4,096장 재생의 [27일 결과](docs/experiments/군집_검수_동일예산_재학습_20261127.md)는 별도 실험이다. 이번 실험과 수치·가중치를 섞지 않는다.

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

이 대조용 추론·시각화 명령의 기본값은 원래 M v6다. 현행 개발 인지는 `ML_ACTIVE_PERCEPTION_RUN` 또는 명시적인 `--adaptation-run`과 해당 버전의 품질 번들을 함께 사용한다. [현재 버전 검사](src/quality_estimation/active_perception.py)는 등록된 가중치·계보를 확인한다. X/y 생성·회귀 학습·RGB 단독 품질 추론 CLI도 사용할 수 있다.

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

버전별 X/y 생성, GP/RF 적합, RGB 단독 검수와 사후 GT 연결은 [데이터·특징·정답 가이드](docs/guides/데이터_등록과_특징정답.md)와 [표시 가용성·누적 갱신 재현](scripts/experiments/scene_cycle_20261128/README.md)을 따른다. [직전 주기 재현](scripts/experiments/environment_cycle_20261127/README.md)은 보존한다. 특징 순서는 각 번들 manifest가 정본이다. 이전 f0의28열·f1의52열과 현재 개발 후보74열은 인지 신원·목표·정책을 확인하고 사용한다.

모델·RGB예제를 포함한 로컬 배포 묶음을 `ML_RELEASE_ROOT`로 지정하면 먼저 아래 검증을 실행할 수 있다. 연구 후보 묶음의 정상 실행은 채택 통과와 다르다.

```bash
python3 scripts/verify_environment_release.py --root "$ML_RELEASE_ROOT"
```

초기 `build_dataset` CLI와 X/y 설계는 [이력](docs/archive/X_y_데이터셋.md)으로 보존한다. 현재9영역 번들의 시작점으로 사용하지 않는다.

## 문서와 데이터 관리

- 공통 설계·평가·역할·개발 방법은 이 저장소의 `docs/`에 둔다. 미확정 설계와 측정된 결과를 구분한다.
- 강의 원문·개인 분석·과목별 발표 준비는 저장소 밖 로컬 워크스페이스에서 관리한다.
  팀 개발에 필요한 결론은 공통 문서에 반영해 로컬 자료 없이도 읽을 수 있게 한다.
- 원본 데이터·영상·마스크·feature·체크포인트·실험 출력은 Git에 올리지 않는다.
- 개인 경로와 컨테이너 이미지 이름은 Git에서 제외한 `config/local.env`에 기록한다.
- 코드 출처·revision과 공개 가능한 재현 근거는 [이관 기록](docs/archive/외부코드_선별.md)에 남긴다.
