# 원거리 보조·보정 지원·환경 전이 근거

[후속 실험 보고](../../../experiments/ROI9_원거리보조와_환경전이_20261126.md)의 집계다. 주 품질 GP의28열·가중치와 인지모델 f0는 그대로다. 원거리 보조31열은 추가 진단 후보이며 전체 IoU용 입력이나 자동 승인 정책을 바꾼 것이 아니다.

| 파일 | 범위 |
|---|---|
| [후보 비교](far_feature_comparison.json), [모델 동결](far_frozen_models.json) | 같은2,480장·3지리fold·GP/RF 각각12설정·세 입력 구성 |
| [보정 동결](calibration_freeze.json), [촬영·동결](collection_and_lock.json) | 새 보정96장 보강, 모델·192장 보정 고정 뒤 새4링크96장 촬영 |
| [새96장](prospective_far_metrics.json), [이전96장 진단](previous96_diagnostics.json) | 원거리 recall의 점 오차·주변 구간·결합 확률. 이전 자료는 미사용 평가가 아님 |
| [주 품질 GP/RF](primary_quality_control.json) | 실제 전체 IoU용28열 RF 대조. 원거리 RF31과 구분 |
| [위치별 짝 차이](paired_far_effects.json) | 4개 위치 단위 bootstrap. 같은 지도의 작은 평가 |
| [환경 전이](environment_transfer.json), [군집 지원](environment_support.json) | 고정v2 RGB 환경 표현의432장 전이와 보정192장의 지원. 환경 임계값을 새로 적합하지 않음 |
| [보정 예산·근거리 보존 진단](environment_calibration_budget.json) | 16군집×정상20장은 최소320장.192장에서는 정상 필터 전에도 적격4군집, 실제Q≥0.7은75장/적격0군집. 근거리 보존을 운행 정상 라벨로 취급하지 않음 |
| [후속 프로토콜](protocol.json), [파일 영수증](receipt.json) | 비교 범위·제한과 공개 파일 해시 |

[14 원거리 추정](14_far_followup.png)은 도로마다 모델 순위가 바뀜을, [15 보정 절충](15_calibration_support_tradeoff.png)은 포함률 이득과 폭 증가를, [16 환경 전이](16_environment_transfer_support.png)는 태그 성적과 검수 기준 지원 부족을 보여준다. SVG·프레임별 출력·원본4채널·모델은 Git 밖 실험 볼륨에 보관한다. UI 스크린샷은 공개 근거에 포함하지 않는다.
