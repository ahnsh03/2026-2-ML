# 검증한 배포 모델

팀원이 pull 후 같은 인지 가중치로 추론·X/y 가공을 시작하도록, 사용자가11월28일 명시한 범위에서 **검증된 배포 가중치만** 관리한다. 원본 데이터·optimizer·전체 replay/학습 원장은 포함하지 않는다.

현재 [r2 체크포인트](perception/m-v6-r2-20261128/checkpoint.pt)는 약2MB이며 Git LFS가 필요 없다. [release manifest](perception/m-v6-r2-20261128/release.json)의 가중치는 기존 r2와 **바이트까지 동일**하다. [등록부](../config/perception_releases.json)의 manifest/가중치/계보 해시를 모두 검사하고 다른 품질 번들과의 혼합을 거부한다.

[현재 실행 카탈로그](catalog.json)는 인지와 기존 검증된 GP52/RF74·환경 번들을 연결한다. 번들은 약50MB이며 가장 큰 파일은49MB 미만이다. 원본 frozen bundle과 바이트까지 같고 개인 경로/비공개 식별자가 없는지 JSON·NPZ 문자열을 검사했다. `adoption.json`은 별도 채택 근거이며 원본 번들의 연구 단계 플래그를 변경하지 않는다. 이는 현재 재현 기준이며 RF/GP의 최종 선택은 대규모 선별·재학습 비교 뒤로 미룬다.

`examples/`의 한 장은1280×720 전방 RGB 실행 예제이며 전체 데이터셋이나 GT를 포함하지 않는다. 모든 런타임 파일은 일반 Git으로 내려받는다.

```bash
PYTHONPATH=src python3 -m quality_estimation.active_perception
```

인지 추론용 의존성은 [40/50 개발 환경](../docs/guides/개발환경.md)을 따른다. X/y 추출과 RGB 검수의 `--perception-release models/perception/m-v6-r2-20261128`은 개인 학습 폴더 없이 이 체크포인트를 사용한다. 원래v6 `--checkpoint`와 외부 검증 학습 원장의 `--adaptation-run`은 별도 대조/연구 경로로 유지한다.

이 폴더는 전체 추가학습 run이 아니다. 새 인지 재학습은 기존31,121장 구성과 중복/역할 검사를 보존한 외부 원장을 준비하고 `--parent-adaptation-run`을 사용한다. 배포 파일만으로 replay를 생략한 학습을 시작하지 않는다. [검증 성적·범위](../docs/baselines/scene_cycle_20261128/confirmation/README.md)

아키텍처 출처는 TwinLiteNet+이며 [MIT 라이선스 고지](LICENSE.upstream)를 함께 보존한다. 모델은 프로젝트에서 재학습한 가중치다.
