# MORAI 카메라·통신 설정

현재 실험 대상은 **26.R1**이다. `network_set_local.json`은 26R1 구조의 로컬 UDP 가져오기 양식이다. 컨트롤러 항목은 MoraiCmdController `PUBSUB_TYPE=771`, `EgoCtrlConfig._PUBSUB_TYPE=771`, JSON `egoCtrlMode=3`으로 정리했다. 원래 내보내기 파일의 TrajectoryTracking `786`을 현재 UDP 명령 송신기의 메시지 형식으로 해석하면 안 된다. 이전 내보내기 내용은 Git 이력에 남는다.

JSON의 운전 모드와 UDP `#MoraiCtrlCmd$` 패킷의 `control_mode=2`는 서로 다른 열거형이다. 패킷 값을 JSON에 그대로 옮기지 않는다. 내부 `/Ego_topic`에 연결하는 상태 입력은 26R1 프로파일 `770`, 이전 프로파일 `295`다. 이전 `295`는 원래 CompetitionInfoPublisher이며 내부 토픽 이름과 시뮬레이터 발행기의 의미를 구분한다. [버전 대응표](../simulator_release_profiles.json)

검토한 26R1 제어 양식은 **payload 27바이트·전체 UDP 59바이트**이며 rear-steer 칸을 포함한다. 내부 beta 메시지에 rear-steer가 없으면 0을 보낸다. 이전 프로파일은 23·55바이트다. 가져오기 양식의 수신 버퍼도 각각 59·55로 맞춘다. 브리지와 launch 기본값을 버전에 따라 분기했고 실제 송신 패킷의 크기·열거형·부호는 네트워크 전송 없이 단위 검증했다.

내부 `/ctrl_cmd`의 `morai_msgs/CtrlCmd.steering`은 rad 단위로 사용하고, UDP 경계에서 최대 조향각으로 나누어 정규화한다. 26R1 프로파일은 송신 조향 부호를 한 번 반전한다. 이전 버전은 `simulator_release:=molitcomp03`으로 선택한다. `front_steer`·`rear_steer` 필드도 기존 브리지가 지원하지만, 현재 저장소의 내부 메시지 계약은 `steering`이다. 시뮬레이터 직접 ROS 연결의 메시지 버전·토픽 대응은 별도 인수 대상이다.

```bash
roslaunch data_collection morai_udp_bridge.launch simulator_release:=26r1
```

이 명령은 브리지를 켜는 예시다. 현재 인스턴스에 네트워크 설정을 자동으로 가져오거나 제어 명령을 발행하지 않는다. 전환 후 포트·실제 메시지 타입·정지 명령·저속 좌우 회전과 상태 피드백의 부호를 확인해야 한다. 상태 `wheel_angle`은 시뮬레이터가 보낸 deg 값을 그대로 유지한다. 26일 Windows UDP 저속 시험에서는 내부 좌회전 양수 명령을 전송에서 한 번 반전했을 때 앞바퀴각·yaw가 양수로 변하고 우회전은 음수로 변했다. 상태 각은 자동 반전하지 않는다. 이 결과는 실측 전송 경로에 해당하며 ROS 토픽 전체의 인수가 아니다. 26R1의 옛 Detail 입력은 기본적으로 비활성화한다. 다른 패킷 크기로 덮어쓸 때는 현재 인스턴스에서 별도로 확인한다.

카메라 ROI는 [9영역 고정 마스크](../roi9_front_v1/profile.json)를 사용한다. 26R1 정지 수집에서 해상도와 보닛 영역의 정합은 점검했지만, 이것이 모든 센서 외부 보정값의 동등성 검증을 뜻하지는 않는다. 기존 자료의 버전 출처는 이전 버전으로 유지하고 신규 자료에 `simulator_release`를 남긴다.

## 11월26일 저속 실측 범위

목표4km/h의 새 구간 주행을 Windows loopback UDP 제어/상태와 native gRPC 환경·4채널 저장으로 수행했다. 명령은27/59바이트, Ego는216/245바이트를 사용했다. 측정한 위치 이동과 native 속도 적분의 비율이 약3.59여서 이번 native/UDP 수집기는3.6으로 나눠 m/s를 기록한다. **ROS 속도 토픽까지 일괄 나누지 않는다.** 전체 경로·직접 ROS 토픽·충돌 피드백의 인수는 별도다.

저장 RPC 미응답이 제어를 붙잡지 않게 비동기로 분리하고 별도 Windows relay의0.6초 heartbeat watchdog을 사용했다. 카메라 노출 시각은 미제공이며 파일 최초 관측은 동기화 시각이 아니다. [주행 근거·제한](../../docs/experiments/ROI9_저속주행과_검수보정_20261126.md) · [수집·검증 도구](../../scripts/experiments/roi9_drive_20261126/README.md)
