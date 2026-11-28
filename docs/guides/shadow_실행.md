# 26R1 RGB shadow 실행

현재 인지 기준은 레포에 포함된 `m-v6-r2-20261128`이다.9영역 특징으로 전체 이미지 IoU를 추정하고, 고정 개수의 버퍼에서 검수 후보를 선별한다. RF와GP의 결과를 함께 기록한다. **조향·가감속과 자동 재학습 편입은 실행하지 않는다.** 온라인에서 GT나 ROI별 IoU를 계산하지 않는다.

## 처음 한 번 준비

인지 런타임 이미지는 [개발 환경](개발환경.md)의40/50 프로필 중 자신의 GPU에 맞는 것을 준비한다. 가중치와 품질 번들은 `git pull`로 받으며 외부 개인 체크포인트 경로가 필요하지 않다. `config/local.env`에 결과 드라이브의 `ML_SHADOW_ROOT`를 설정한다. 데이터·로그는 그 경로에 저장한다.

```bash
# config/local.env에 자기 PC 경로로 지정
ML_SHADOW_ROOT="/path/to/outputs/shadow"
ML_GPU_PROFILE="auto"  # 또는 rtx40 / rtx50
```

다음 명령은 포함된 RGB 한 장으로 모델·선별·마지막 짧은 버퍼를 확인한다.

```bash
./scripts/shadow.sh demo
```

40프로필의 CPU 의존성 확인은 아래와 같다. **이 결과는40실기 GPU 검증을 대신하지 않는다.**

```bash
ML_GPU_PROFILE=rtx40 ML_SHADOW_DEVICE=cpu ./scripts/shadow.sh demo
```

40계열 실물 GPU에서의 인계 검사는 CPU 옵션 없이 실행한다. 출력 폴더의 context.json에 실제 GPU 이름·capability와 번들 해시가 남아야 한다.

```bash
ML_GPU_PROFILE=rtx40 ./scripts/shadow.sh demo
ML_GPU_PROFILE=rtx40 ./scripts/shadow.sh replay "$SHADOW_SESSION" --processed-only
```

세션 재생에는 해당 PC에서 접근 가능한 보존 RGB 원본이 필요하다. 실제40GPU가 없는 PC의 성공 결과를 실물 재현으로 기록하지 않는다.

## ROS 기본 연결

```bash
./scripts/shadow.sh bridge
```

이 명령은 전용 ROS bridge 컨테이너를 준비한다. 최초 실행은 `docker/shadow-ros/Dockerfile` 빌드가 필요할 수 있다. 무거운 인지 학습 환경과 ROS1 Python 환경은 별도 프로세스로 실행한다.

시뮬레이터 전방 카메라 설정을 확인한다.

- 카메라1,1280×720, 화각90°, 위치(1.9,0,1.2), pitch2°의 고정 배치. 다른 카메라 영상을 전방으로 연결하지 않는다.
- F5 Sensor List의 전방 **Capture Mode를 끈다.** Capture Mode와 지속 영상 송신은 별도 동작이다.
- F3 전방 카메라 설정에서 ROS를 선택한다. bridge IP는127.0.0.1, port9090, topic `/image_jpeg/compressed`, frame ID `Camera-1`로 설정하고 Connect한다.
- 시뮬레이터/bridge를 다시 시작했으면 Connect가 다시 필요할 수 있다. `stop`은 bridge를 유지하므로 보통 이후 세션에는 재설정이 필요하지 않다.

[MORAI Capture Mode 설명](https://help-morai-sim-en.scrollhelp.site/morai-sim-standard-en/sensor-list)에 따라, 동기 RGB/Semantic/Depth/Instance 수집으로 돌아갈 때에는 스트림을 끄고 Capture Mode로 전환한다.

## 반복해서 실행하는 명령

```bash
./scripts/shadow.sh start                    # 10분, 초당1장,100장 버퍼
./scripts/shadow.sh status                   # 수신·추론·생략·선별 상태
./scripts/shadow.sh stop                     # 수신 종료, 마지막 버퍼 선별
./scripts/shadow.sh replay /path/to/session  # 보존한 RGB 전체를 재생
./scripts/shadow.sh replay /path/to/session --processed-only  # 실시간 처리 모집단·추론 배치까지
```

기간·수집률·버퍼는 필요한 경우만 지정한다.

```bash
./scripts/shadow.sh start --seconds 600 --hz 1 --window 100
```

`start`는 모델을 먼저 초기화·예열하고 수신기를 시작한다. 입력이 오지 않으면 접속과 Capture Mode를 확인한다. 마지막 부분 버퍼도 동일한10% 예산을 올림해 처리한다. 따라서 한 장짜리 버퍼는 한 장을 남긴다. 선별률은 창별 올림과 시간 표본화의 영향을 받는다.

`status`에서 원본 수신량과 실제 추론량을 별도로 본다. 추론이 밀릴 때에는 최신 batch를 처리하고 건너뛴 ID를 기록한다. 해당 RGB는 원본 보관함에 남으므로 나중에 전체 재생할 수 있다. GPU에서20Hz를 모두 처리한다는 의미가 아니다.

## UDP 대체 경로

현재 PC의 Windows 송신을 그대로 사용할 때에는 전방 카메라를 UDP source9290→destination127.0.0.1:9291로 연결하고 다음을 실행한다.

```bash
./scripts/shadow.sh start --transport udp
```

이 경로에는 Windows `py.exe`와 Windows에서 접근 가능한 결과 드라이브가 필요하다. UDP 조각의 순서·동일 stamp·크기·JPEG·전방 해상도를 검사한다. 조각이 빠지거나 다른 프레임이 섞이면 버리고 집계한다. [공식 UDP 예제](https://github.com/MORAI-Autonomous/MORAI-SensorExample)를 참조한다. UDP와 ROS 수신기를 같은 카메라에 동시에 켜지 않는다.

## 남는 기록과 해석

| 위치 | 용도 |
|---|---|
| `archive/rgb/`, `rgb.jsonl` | 세션 UUID를 포함한 전역 프레임 ID, 원본 JPEG와 SHA, 카메라·수신 시각·native/ROS stamp. 재가공 입력 |
| `archive/receiver_*.json` | 수신량, 의도한 표본화, 시퀀스 누락/리셋, 저장 한도와 종료 이유 |
| `review/context.json` | 인지·품질 번들·ROI·특징 순서·코드·런타임·선별 창의 고정 정보 |
| `review/predictions.jsonl`, `features.npz` | GT 없는 예측과 특징. RF/GP 결과를 분리 기록 |
| `review/candidates.jsonl` | 모델별 검수 ID·순위·선별 이유·감사 표본 확률 |
| `review/inference_skipped.jsonl` | 원본은 보존했으나 실시간 추론에서 생략한 ID |
| `review/processed_rgb.jsonl` | 실시간에 처리한 정확한 모집단. 같은 모집단의 오프라인 대조에 사용 |
| `review/run.json`, `_SUCCESS` | 완료 상태와 해시. 입력0장은 성공 근거로 삼지 않음 |

replay는 원래 세션의 버퍼 크기·예산을 이어받는다. 전체 원본 재생에는 실시간 생략 프레임도 포함되므로 선택 ID가 달라질 수 있다. 정확한 실시간 선별 대조에는 --processed-only를 사용하며 영상은 복사하지 않는다.

센서 stamp를 저장했다는 이유로 카메라 노출 시각이나 UTC 동기 오차를 검증한 것으로 해석하지 않는다. 처리 시간과 원본 수신 이후 대기 시간도 분리한다. 단일 ROI 오류 점수와 GP z는 실패 확률이나 안전 판정이 아니다.

이 명령은 최대 저장량1GiB·여유 공간5GiB를 기본 한도로 사용한다. 시뮬레이터 차량 상태나 날씨를 변경하는 코드는 포함하지 않는다. 주행은 별도의 검증된 제어기로 실행하며 shadow 결과를 제어 명령에 연결하지 않는다.


## 확인된 범위와 재생 수치

26R1 ROS300초/저속6구간에서293장을 저장·추론했고 처리 평균0.0808초/프레임이었다.20Hz 입력을1Hz로 표본화한 결과이며 최대 처리량이나 종단 지연 보장은 아니다. 창100/100/93장마다10/10/10개를 각 번들에서 선별했다. [결과·한계](../experiments/대비특징_검수와_shadow_20261129.md)

일반 replay는 보존 RGB 전체를 기본 배치로 처리한다. --processed-only는 원래 처리 ID·창·추론 배치도 보존한다. GPU 배치가 바뀌면 작은 특징 차이가 RF split을 바꿀 수 있어 엄밀한 재생에는 후자를 쓴다.293장과 최종 소스146장 모두 특징/예측 차이0·선택 ID/순서/사유 일치였다.

기존 bridge가 다른 checkout/ML_SHADOW_ROOT를 마운트했으면 시작 전에 거부한다. 원래 설정으로 실행하거나 별도 ML_SHADOW_ROS_CONTAINER를 지정하고 그 bridge로 카메라를 연결한다. 다른 세션의 원본 경로로 조용히 저장하지 않는다. 실제40계열 GPU는 아직 미검증이며 팀원 장비에서 demo·세션 재생이 필요하다.


기본10분 실행도586장 전체 추론·생략0·sequence gap0으로 종료했다. 수신 콜백→점수 기록 P95는0.2024초였고 시뮬 노출/전송 지연은 제외한다. [10분 영수증](../baselines/shadow_cycle_20261129/reports/endurance_10min.json). --hz1은 최대 저장률이며 실제20Hz 송신과 간격 조건 때문에 정확히600장을 저장하는 의미는 아니다.
