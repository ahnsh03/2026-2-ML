# 11월26일 저속 주행·특징·공간 보정 근거

[결과와 최종 판정](../../experiments/ROI9_저속주행과_검수보정_20261126.md) · [실행 도구](../../../scripts/experiments/roi9_drive_20261126/README.md) · [수치 집계](results.json) · [동결 실행 구성](runtime_manifest.json) · [검증 범위](verification.json)

**주 품질 입력은 9영역·28열 GP다.** 원거리31열은 새 확인의 추가 MAE 이득이 사전 기준보다 작아 기본 대체로 채택하지 않았다. 별도28+31 진단 번들의 보정 v2는 근/원거리 구간·공간 확률을 연구용으로 제공한다. 기본 검수는 공통 IoU0.7의 저품질 확률순이며 환경별 하향 임계·공간 할당·자동 학습 승인은 채택하지 않았다.

## 그림 바로가기

| 설명 | 파일 |
|---|---|
| 현재9영역 | [1280×720 ROI](roi9_native.png) |
| 특징 유지/제거 | [개발 비교](main_01_feature_selection.png) |
| 새 확인 MAE·보정·구간 폭 | [집계](confirmation_01_fresh_confirmation.png) |
| 실제 궤적·속도 | [새3구간](drive_actual_motion.png), [처음9구간](motion_06_actual_motion.png) |
| heartbeat 중단 | [정지 관측](motion_07_watchdog_stop.png) |
| GP의 큰 과대추정·근거리FP | [c04 주야 사례](confirm26_c04_day_night.png), [사후count](c04_falsepositive_diagnosis.json) |
| 원거리recall만 저하 | [GT선택 설명 사례](probability_class_2.png) |
| 희소GT의 한계 | [원래 주평가 사례](main_sparse_markings.png) |

## 집계의 자료 역할

| 집계 키 | 자료와 해석 |
|---|---|
| `primary_test` | 원래 고정 후 평가3구간·366후보. 차선 GT 희소 구간을 포함한다. |
| `supported_followup` | GT 지원만으로 고른 후속3구간·368후보. v1 비교 후 v2 보정용으로 전환했으므로 v2 미사용 평가가 아니다. |
| `fresh_confirmation` | v2·특징·정책 고정 후 별도3구간·366후보 확인. 원거리31열 대체 기준은 미달했다. |
| `confirmation_adoption` | 채점 전 기록한 기준과 통과/보류. 보정 진단 기준 통과가 전체 시스템 채택은 아니다. |
| `current_data_roles` | 전역28,971행의 현재 노출 이력. 원래 획득 역할을 보존하고 후속 전체 구간의 보정 관련 노출을 별도로 남긴다. |

모든 MAE는 IoU/recall의0~1 단위다. 주 비교는 도로별 동일 가중 평균이며 반복 날씨·인접 프레임을 별도 도로로 세지 않는다. 도로별 전체 후보10% 올림 예산39슬롯과 세션/64프레임 창 재생45슬롯은 다른 비교다. 낮은 IoU와 운행 실패, 데이터 선별과 인지 재학습 성과를 구분한다.

## 그림 선택

| 목적 | 파일 |
|---|---|
| 현재9영역 실제 주행 사진 | [roi9_native.png](roi9_native.png) |
| 주28열 유지와 원거리 보조 개발 비교 | [특징 실험](main_01_feature_selection.png) |
| 새 도로의 최종 수치·구간 폭·공간 확률 | [확인 결과](confirmation_01_fresh_confirmation.png) |
| 실제 주행 위치·속도·추종 오차 | [9구간 지도](motion_06_actual_motion.png), [확인3구간 상세](drive_actual_motion.png) |
| heartbeat 중단 실측 | [watchdog](motion_07_watchdog_stop.png) |
| 희소 차선 GT의 IoU 해석 | [원평가 사례](main_sparse_markings.png) |
| 낮 장면에서도 품질 GP가 높게 오판한 사례 | [확인 구간 A](confirm26_c04_day_night.png) |
| 표시 차선 도로의 주야 비교 | [후속 구간 B](support26_c02_day_night.png), [확인 구간 C](confirm26_c07_day_night.png) |
| 근거리 유지·원거리 약화의 분포 | [공간 사례](probability_class_2.png) |
| 두 띠 유지·두 띠 약화 | [유지 사례](probability_class_0.png), [약화 사례](probability_class_3.png) |

과학 그림·사진 조합은2560×1440(16:9), ROI 사진은1280×720(16:9)이다. 과학 그림은 같은 이름의 SVG도 제공한다. 사진은 각 행의 실제 RGB/Semantic·동일 인지모델 출력으로 구성했고 프레임별 해시를 검증했다. 주야 사진의 Ego 위치 차이는 그림에 표시한다. 노출 동기화·완전히 같은 자세·가시성 기준의 정상/실패를 증명하는 그림은 아니다.

GT로 고른 공간 사례는 설명용이다. 공간 상태는 recall 누락 사건이다. c04 낮 장면은 근거리recall0.929여도 precision0.463·근좌FP2,475픽셀이므로 ‘근거리 정상’으로 해석하지 않는다. GT연결 결과에는 precision·FP수를 별도로 제공한다.

근거리recall만 약한 상태는 확인 자료에0장이므로 그 상태의 사례나 검증 성과를 만들지 않았다. 포함률 개선은 구간 폭 증가와 함께 읽고, v2의 Brier가 원시 분포보다 조금 나쁜 점도 함께 본다.

전체 원본4채널·원해상도 사진 성분·실행 기록·동결 가중치·학습 구성원·사후 GT는 실험 볼륨에 보존한다. `test_not_collected` 같은 동결 JSON의 플래그는 생성 당시 상태이며 현재 완료 상태는 이 문서와 검증 집계로 판단한다. [파일 해시](artifact_sha256.json)는 공개 그림과 집계의 전송 대조용이다.
