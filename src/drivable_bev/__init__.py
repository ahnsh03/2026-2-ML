"""카메라 → BEV 투영·융합 (인지 코드 레포 d0922bb `drivable_bev`에서 ML에 필요한 모듈만 이관).

원본 패키지의 `__init__`은 차선 추적·경계 후보 등 15개 모듈을 함께 불러오므로 옮기지 않았다.
하위 모듈을 직접 import한다: `from drivable_bev.calibration import ...`
"""
