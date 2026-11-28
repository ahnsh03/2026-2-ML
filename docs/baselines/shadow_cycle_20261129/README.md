# GP 대비·새534장·ROS shadow 근거

[결론/설계](../../experiments/대비특징_검수와_shadow_20261129.md) · [실행](../../guides/shadow_실행.md) · [재현 소스](../../../scripts/experiments/shadow_cycle_20261129/README.md)

기준은 `m-v6-r2-20261128`·9영역이다. **z 순위 결합과 GP57 입력 모두 제외**, RF/GP 병행·r2 유지다. 이번 단위의 신규 인지 학습은 없다. 정확도 확인534장과 GT 없는 주행 기능 확인을 구분한다.

| 근거 | 파일 |
|---|---|
| 결정·native 수신/주행 | [decisions](reports/decisions.json) |
| 모든 고정 선별·무작위 대조 | [confirmation](reports/confirmation.json) |
| 같은 학습량 OOF 입력 | [개발](reports/quality_input_development.json), [새 확인](reports/quality_input.json) |
| 고정90% 구간의 전이 | [시간대별 포함률](reports/fixed_interval_diagnostic.json) |
| 새벽/가시성·예산 | [QA](reports/visibility_qa.json), [예산](reports/review_budget.json) |
|293장·최종 소스146장 재생 | [주행](reports/moving_exact_replay.json), [최종 소스](reports/final_source_exact_replay.json) |
| 입력 거부·과부하·UDP | [통제 프로토콜](reports/transport_boundaries.json) |
| 동일 위치 시간 전환 | [RGB 밝기 진단](reports/photometric_probe.json) |
| 기본10분·원본/처리 집계·자원 | [586장 감사](reports/endurance_10min.json), [자원](reports/endurance_resources.json) |
| 주행 수신→점수 시각 | [293장 감사](reports/moving_archive_audit.json) |
| 최종 차량 원상태 관측 | [Keyboard/Park·위치·Sunny13](reports/final_restoration.json) |
| 코드 검사 | [353개/프로필](reports/code_validation.json) |
| 실제 게시 파일 해시 | [publication](publication.json) |

## 그림

- [01 GP 대비 새 확인](figures/01_contrast_confirmation.png)
- [02 GP52/GP57/RF52 입력](figures/02_quality_input_confirmation.png)
- [03 예산 상한·선택 정밀도](figures/03_review_budget_context.png)
- [04 새벽 미회수·가시성 cue](figures/04_dawn_visibility_qa.png)
- [05 실제 ROS 주행](figures/05_native_ros_shadow.png)
- [06 주차 위치 시간 전환](figures/06_photometric_settling.png)

SVG도 같은 폴더에 있다. [생성 입력/소스 해시](figures/manifest.json)를 보존했다.06번 해시는 photometric report/publication에 있다. 독립 확인의 물리 구간은6개다.

## 실제 RGB·GT·예측·오류

[원장](gallery/manifest.json)의9개 중4개를 공개한다. 나머지는 로컬 보존이다. RGB16:9·모델 격자640×360을 유지했고 표시 GT/예측의 TP/FP/FN이 점수 팩과 정확히 같다. 각 사례 폴더에 개별 `rgb.png`, `gt.png`, `prediction.png`, `error.png`도 있다.

- [01:05시 첫 미회수 FN](gallery/case_01.png), [RGB](gallery/case_01/rgb.png)
- [04:06시 첫 미회수 FP](gallery/case_04.png), [RGB](gallery/case_04/rgb.png)
- [07:근거리 유지·원거리 오류](gallery/case_07.png), [RGB](gallery/case_07/rgb.png)
- [08:동일 위치13시](gallery/case_08.png), [RGB](gallery/case_08/rgb.png)

FP/FN은 동시에 발생할 수 있다. 이 그림으로 물리 가시성 정상 여부를 판정하지 않는다. 전체 raw·optimizer·미채택 GP57은 포함하지 않는다. [검증 배포 모델](../../../models/README.md)은 별도다.40실기 GPU는 아직 미검증이다.
