# 26R1 저속 주행·9영역 특징·공간 확률 검증

이 스크립트는 실제 저속 주행에서 자료를 보존하고, 개발·보정·미사용 평가 구간을 분리해 고정 GP와 검수 후보를 검사한다. 인지모델은 Medium v6 EMA를 유지한다. 데이터·모델·원시 로그는 실험 볼륨에 저장한다.

## 실행 순서

| 순서 | 도구 | 출력·제약 |
|---|---|---|
| 1 | `plan_routes.py` | 현재 지도와 이전 촬영 위치를 대조한 구간 계획. 같은 지도 안의 새 구간이며 전체 링크·인지 학습 장면의 미노출을 보장하지 않는다. |
| 2 | `windows_udp_relay.ps1`, `collect.py` | Windows loopback UDP 제어·차량 상태·0.6초 watchdog, native gRPC 환경/배치/4채널 저장. 명령 조향은 좌회전 양수, 전송 부호는 한 번 반전한다. |
| 3 | `register.py` | 완결 주행의 원본4채널 해시·이미지·색상·고유ID/픽셀·실제 위치 검증. 전체 목록과3m 간격 목록을 분리한다. |
| 4 | 이전 실험의 `extract.py` | 같은9영역372후보와 별도 GT. 주/보조 입력을 바꿔도 원본을 다시 가공할 수 있다. |
| 5 | `feature_study.py` | 기존 동결2,480장·동일 지리 CV에서 GP/RF 적합. 새 개발 자료만으로 특징 후보를 판단한다. |
| 6 | `calibrate.py`, `review_policy.py` | 별도 보정 구간의 도로 제외 검증. 경험적 평균/분산·상관·네 상태 확률, RGB 환경 참조, 동일 예산 BO/무작위 대조. |
| 7 | `score.py` | 모델·특징·보정·정책 고정 영수증 이후 촬영한 주평가. 학습·튜닝 금지. 정답 지원 부족은 성공으로 집계하지 않는다. |
| 8 | `score_condition_shift.py` | 미리 선언한 추가 조건의 같은 평가 구간. 새로운 도로의 반복 검증으로 부르지 않는다. |
| 9 | `audit_dataset.py`, `alignment_audit.py`, `merge_registry.py` | 원본 재검증·역할 간 중복/거리·RGB/GT 정합·전역 재가공 목록. |
| 10 | `make_figures.py`, `prepare_rgb_replay.py`, `case_figures.py` | 16:9 PNG/SVG와 프레임마다 실제 연결한 GT 사례. 사진 원본은 실험 볼륨에 남긴다. |

각 도구의 `--help`로 입력을 확인한다. 입력 경로는 환경에 맞춰 제공하며 기존 완료 폴더를 덮어쓰지 않는다. 시뮬레이터 제어에는 공급사의 native gRPC Python 정의와 Windows 네트워크 설정이 필요하다. 직접 ROS 메시지·충돌 피드백·실시간 FPS를 인수한 도구는 아니다.

## GT 없는 추론·사후 정답 연결

`package_review_bundle.py`는 고정 특징/GP·보정·정책·학습 출처·RGB8장 묶음을 만든다. 체크포인트와 원본 GT는 포함하지 않는다. 팀 레포와 같은 코드 버전에서 다음 순서로 재생한다. 두 GPU 프로파일은 [개발 환경](../../../docs/guides/개발환경.md)을 따른다.

```bash
PYTHONPATH=src python3 -m quality_estimation.drive_bundle \
  --bundle "$REVIEW_BUNDLE" --simulator-release 26.R1

PYTHONPATH=src python3 -m quality_estimation.infer_review_images \
  --manifest "$REVIEW_BUNDLE/rgb_manifest.jsonl" --data-root "$REVIEW_BUNDLE" \
  --feature-study "$REVIEW_BUNDLE/feature_study" \
  --calibration "$REVIEW_BUNDLE/calibration" \
  --policy "$REVIEW_BUNDLE/policy/policy_frozen.json" \
  --checkpoint "$PERCEPTION_CHECKPOINT" --simulator-release 26.R1 \
  --device cuda --output "$NEW_REPLAY"

PYTHONPATH=src python3 -m quality_estimation.review_calibrated \
  --replay "$NEW_REPLAY" --output "$NEW_QUEUE" \
  --window-size 64 --budget-fraction 0.1
```

40프로파일 CPU 검증에서는 `--device cpu`를 쓴다. 이는 실제40 GPU 검증을 뜻하지 않는다. 추론 입력은 RGB와 인지 출력이며 GT·위치·날씨/시각 설정은 넣지 않는다. 큐는 세션 경계와 동점을 고정하고, 창별 올림 때문에 전체 비율이10%를 넘을 수 있다. 주평가의 도로별 전체 후보10% 예산과 구별한다. `verify_replay.py`는 재가공 특징·두 프로파일을 대조하고 큐 확정 후에만 GT를 연결한다. 모든 후보의 `admission_approved`는 false다.

## 결과 해석

전체 흰/노란 차선 IoU 하나가 주 정답이다. 근/원거리 **영상 띠**의 recall은 오프라인 보조 정답이며 온라인 ROI IoU를 계산하지 않는다. 근거리0.8·원거리0.5 기준의 네 상태는 `both_good`, `near_weak_only`, `far_weak_only`, `both_weak`다. 서로 배타적인 확률의 합은1이며 검수 `priority`는 확률이 아니다.

원거리 별도 GP와 주 GP의 초기 공분산을0으로 합성하고 경험적 잔차 상관을 별도로 비교한다. 이를 학습된 하나의 정확한 다중 출력 GP 사후분포라고 부르지 않는다. 작은 Brier 차이·정답이 없는 상태·좁은 구간 포함률·표시 차선이 거의 없는 도로는 보고서에서 구분한다. 환경별 참조는 운행 정상/가시성 정답이 아니다. 모델/정책·인지 재학습·라이브 ROS 채택은 서로 다른 판정이다.
