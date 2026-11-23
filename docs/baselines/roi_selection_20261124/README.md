# ROI 기본값 결정 근거

결론과 해석은 [7영역·10영역 비교](../../experiments/ROI_7영역_10영역_비교_20261124.md), 이후 작업은 [ROI 가이드](../../guides/ROI와_특징추출.md)를 따른다. 기본 프로파일은 7영역, 시작 특징은 전체 화면 통계를 포함한 96열이다. 이 폴더는 당시 비교의 공개 집계와 그림을 담는다.

| 파일 | 용도 |
|---|---|
| [protocol.json](protocol.json) | 원래 비교 계획과 두 추가 수집의 시점·조건·보존 해시, 판정 해석 범위 |
| [reproduction_protocol.json](reproduction_protocol.json) | 비교 스크립트가 읽는 그룹·시드·3개 구성·RF/GP 각 12개 설정 |
| [development.json](development.json) | 개발 자료에서 고른 구성·설정·차원·모델 해시와 12개 전체 후보 |
| [selected_schema.json](selected_schema.json) | 기본 96열의 이름·순서·마스크·단위 규칙 |
| [results.json](results.json) | 코호트별 전체/정의/저IoU 수, 선택 구성·동일 구성의 차이, 시드·공통 설정 대조 |
| [support_sensitivity.json](support_sensitivity.json) | 차선 GT 픽셀 수에 따른 사후 진단. 전체 자료의 주 성적과 별개로 보존 |
| [benchmark.json](benchmark.json) | 같은 입력에서 잰 CPU 비용, 버전·측정 범위 |
| [verification.json](verification.json) | 학습 재현·원본/정답 검증·코드 스냅샷 해시 |
| [artifact_hashes.json](artifact_hashes.json) | 이 폴더에서 내보낸 파일의 SHA-256 |

그림은 `01_paired_mae.png`(코호트별 오차), `02_paired_uncertainty.png`(공간 묶음 차이 구간), `04_roi_geometry.png`(같은 이미지의 마스크), `05_exact_gt_pair.png`(같은 위치·날씨의 낮/밤 RGB·정확히 대응하는 GT·v6 예측)다. 원본 해시와 그림 생성 코드는 실험 볼륨에 보존한다.

전체 데이터·모델·프레임별 예측은 별도 실험 볼륨에서 인계한다. 공개 집계의 모델 해시는 그 원본과 대조하는 값이다. 기존 자료와 두 추가 수집은 이번 선택에 사용했으므로 최종 미사용 도로 평가에 다시 쓰지 않는다.
