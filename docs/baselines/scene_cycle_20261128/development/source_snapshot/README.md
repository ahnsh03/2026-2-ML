# 실행 소스 원문 보존

현재 CLI에 인수·검증을 추가하기 전 실행된 소스를 원래 바이트 그대로 보관한다. 각 파일 SHA는 [공개 보존 목록](../publication_manifest.json)의 `original_sha256` 및 해당 보고서의 `source_sha256`와 일치한다.

- [Q/오류 적합 원문](fit_review_models_executed.py)
- [새 Dev 선택 원문](evaluate_development_executed.py)
- [환경 fit 원문](fit_environment_executed.py)
- [e-means 보강 원문](refine_environment_executed.py)

이 폴더는 과거 실행을 추적하기 위한 증거다. 상대 import를 실행 위치에 맞춰 고치지 않았다. 현재 실행 도구는 저장소의 `scripts/experiments/scene_cycle_20261128/`에 있다.
