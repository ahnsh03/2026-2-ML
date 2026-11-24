# 특징 실험 설정

원래 설정 파일을 그대로 보존했다. `protocol.json`은 초기 의도별 후보·데이터 역할·시드, `matched_budget_protocol.json`은 120·260열의 동등 예산 대조다.

| 단계 | 프로토콜 | 실행 코드 |
|---|---|---|
| 후보·출력의 범위 | `protocol.json`, `output_controls_protocol.json` | `compare_region_candidates.py` |
| 내부 순위 축소 | `reduction_protocol.json`, `reduction_controls_protocol.json` | `reduce_region_candidates.py` |
| 24열 이후 관계 | `relations_protocol.json` | `compare_compact_relations.py` |
| 현재 27열 비교 | `raw_uncertainty_protocol.json` | `compare_compact_relations.py` |
| 원본 확률 단독·넓은 입력 대조 | `uncertainty_context_protocol.json` | `compare_region_candidates.py` |
| 사후 면적 제거 진단 | `coverage_controls_protocol.json` | `compare_compact_relations.py` |
| RGB 환경 후보 | `environment_focus_protocol.json` | `compare_region_candidates.py` |
| 날씨·시각 입력 축소 | `environment_reduction_protocol.json` | `reduce_region_environments.py` |
| e-means/K-means | `cluster_protocol.json` | `compare_region_environments.py` |

실행 파일은 레포 `scripts/`에 있다. 원본 848열 또는 원본 확률 3열이 추가된 851열 후보 팩의 스키마와 ID·해시를 함께 전달한다. 각 명령은 `--packs`, `--protocol`, 새 `--output-dir`와 `--jobs`를 받으며 상세 옵션은 `--help`를 따른다. [추출과 단일 이미지 실행](../../../guides/ROI와_특징추출.md) · [결과와 데이터 역할](../../../experiments/ROI_9영역_특징선정_20261124.md)

프로토콜의 상속된 `created_utc`는 초기 기본 설정 시각이다. 후속 후보의 실제 동결·추론·집계 시각은 [후속 요약](../followup_summary.json)을 따른다. 특징 선택·정책 보정에서 이미 살펴본 도로는 개발·진단 역할을 보존한다.
