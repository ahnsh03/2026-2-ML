# A6. MORAI 데이터 군집 아이디어 모음

> 계열 A (MORAI) · 9/23 신규 · 상태: **A1의 분석 섹션 후보** (단독 주제 아님)
> 고유 오류맵 군집은 A1 본체에 포함 → [A1 구현 계획 §6](../05_구현계획/A1_구현계획.md#6-실패-유형-발견--고유-오류맵-군집)

군집 결과를 이미지로 바로 확인할 수 있는 MORAI 소재를 모아 둔다.

| 아이디어 | 무엇을 발견하나 | 검증 방법 | 판정 |
|---|---|---|---|
| **조건 군집의 숨은 축** | 영상 통계·임베딩 군집이 sunny/foggy × 시간대로 갈리는가, 아니면 **주행 방향(역광 vs 순광)** 같은 다른 축으로 갈리는가 | 조건 라벨(정답 있음)과 ARI. heading과 sim 시각으로 역광 여부 대조 (역광 축은 가설) | A1 가설 H1의 보조 분석으로 흡수 |
| **장면 군집 + 커버리지 감사** | 학습 데이터에 적게 들어간 장면이면서 IoU도 낮은 장면 → 다음 수집 우선순위 | 군집을 K-City 지도 위에 찍어 MGeo 속성(교차로·횡단보도·음영 구간)과 대조 | A1의 장소 분석으로 흡수 |
| LiDAR 객체 군집 | GT 박스 점군(점 수·크기·높이 분포)으로 클래스가 분리되는가 | 클래스 라벨과 대조 | 발견 여지 낮음. capture LiDAR GT 박스 파일 67%가 비어 있고 중앙값 8점, Obstacle 박스는 신뢰 불가라는 기존 결과를 정량화하는 수준 |
| 주행 거동 군집 | `/Ego_topic` 50 Hz로 손 운전 vs 자동주행 스타일 | 컨트롤러 라벨과 대조 | 인사이트 약함 |

## 데이터 메모 `[확정]`

- 프레임마다 RGB·Semantic·Instance·Depth·BBox2D/3D: `C:\MoraiLauncher_Win\SensorData\CAMERA_1~4` (약 1만 frame, 22 GB)
- 같은 프레임 이름의 LiDAR: `C:\MoraiLauncher_Win\MoraiLauncher_Win_Data\SaveFile\SensorData\LIDAR_5` (일부 run은 옛 이름 → `data/run_id_rename_history.jsonl`)
- Capture는 캡처마다 시뮬이 약 0.5초 멈춰 **연속 궤적이 아니다.** 시계열 군집에는 rosbag을 쓴다
