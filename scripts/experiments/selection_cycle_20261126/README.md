# e-means·검수·인지모델 갱신의 통합 비교

진행 중인 연구 코드다. 완료·채택 여부는 결과 문서를 따른다. 기본 인지는 Medium v6 EMA이며, 9영역과 전체 white/yellow 차선 IoU 목표를 유지한다.

## 비교와 데이터 경계

| 항목 | 고정 예산 |
|---|---|
| 추가학습 후보 | 기존 수집 7,442장, 원래 v6 train/val과 decoded RGB 중복 없음 |
| 새 자료 선별 | Random / QualityRank / ClusterBO 각 3,072장 |
| 공통 replay | 원래 v6 train의 4,096장, 전·좌·우 카메라와 수집 run에 걸쳐 분산 |
| 학습 | 각 2시드, f0에서 시작, batch16=새8+replay8, 24,000step |
| continuation 대조 | ReplayOnly: 새8 슬롯을 다른 original train 3,072장으로 대체, 공통 replay8 유지 |
| 정답 | 고정 actor-negative 정책의 drivable/road_marking 2헤드, padding 제외 |

BF16 forward·FP32 loss reduction·고정 BatchNorm 통계·cosine LR·공통 replay 슬롯의 f0 distillation을 사용한다. 새 자료의 학습 투입 비율은 3,072/7,442이며, 과거 10% 온라인 로그 예산과 다르다. 선별은 GT 없는 특징/GP 점수로 수행하고, 기존 Semantic의 검증·정답 생성은 admission 단계에서 수행한다.

`prepare.py` → 공유 `adaptation_cache` → `policy.py` → `orchestrate.py` 순서다. 데이터 루트·원본 dataset 루트·체크포인트 위치는 CLI로 전달한다. `audit_inputs.py`는 전체 원래 노출 RGB와 정답 재생성 일치를 확인한다. 원본 RGB/Semantic을 복사하지 않으며 같은 캐시를 모든 arm이 사용한다.

`amend_roles.py`는 v1을 보존하고 조건 전환 cohort의 접두사 충돌을 수정한다. 가까운 과거 평가 장면을 분리하고, quality fit/dev/cal/heldout 사이 최소25m 경계를 검증한다. 새 자료·quality fit·development의 구성은 변경하지 않는다. 이후 품질 갱신과 최종 평가에는 `source_packs_v2.npz`, `all_sources_v2.jsonl`, `evaluation_cached_v2.jsonl`을 사용한다.

## 군집·BO의 범위

- 공식 [Faiss e-means](https://github.com/facebookresearch/faiss/blob/main/contrib/e_means/README.md)의 `NumpyEmeans`, RGB4종×K8/16/32×3시드를 비교한다. 마지막 학습 pass의 배정 대신 **최종 중심으로 최근접 배정을 다시 계산**한다.
- 군집 특징은 RGB만 사용한다. 날씨·시각 메타데이터는 오프라인 환경 정보량 진단에만 사용하고, 온라인 입력에는 넣지 않는다.
- `policy.py`의 BO는 development의 FP/FN 검수 목적함수를 탐색한다. 같은 초기점·16회 무작위 탐색과 비교하며, 그 목적함수를 재학습 이득으로 부르지 않는다.
- `relative_policy.py`는 보정 도로에서만 환경별 상대 검수 참조를 적합한다. 절대 IoU0.7 판정과 별개다. 지원이 부족하거나 OOD인 경우 공통 기준으로 돌아간다. v1은 역할 충돌을 발견하기 전의 이력이고, v2가 유효한 비교다.
- `auxiliary_study.py`는 가까운 영역의 recall만으로 정상이라 부르지 않도록 near F1 보조 목표를 비교한다. ROI의 GT 지표는 오프라인 학습·평가용이며 온라인 ROI IoU를 요구하지 않는다.

## 검증과 실행 환경

`evaluate_queue.py`는 development와 원래 validation의 동일 픽셀에서만 중간 결과를 확인한다. 최종 진단은 모델·정책 선택을 고정한 뒤 수행한다. 이미 과거에 열었던 simulator 자료이므로 untouched final test나 실제 주행 안전 검증이라고 주장하지 않는다.

`retrain_streaming`은 체크포인트·optimizer·RNG를 `latest.pt`에 보존한다. 완료된 `checkpoint.pt`는 manifest·정책·회원 목록·소스 해시와 `_SUCCESS`를 함께 제공해야 새 perception identity로 읽을 수 있다. 중간 checkpoint는 배포 모델이 아니다.

학습은 RTX50 CUDA12.8 환경을 사용한다. RTX40 CUDA11.8 환경을 유지한다. e-means/BO는 [별도 CPU 연구 의존성](../../../docker/research/requirements-cpu.txt)을 설치한 환경에서 실행하고, 군집은 NumPy 중심·표준화로 내보내므로 추론에 Faiss가 필요하지 않다.
