# ROS1 수집 컨테이너 정의 (Ubuntu 20.04 + ROS Noetic).
# 이관: 인지 코드 레포 d0922bb `docker/ros-noetic/Dockerfile`. 참고·재현용 정의다.
# 평소에는 새로 빌드하지 않고 기존 이미지를 config/local.env 의 ROS_IMAGE 로 지정해 쓴다.
# 빌드한다면 기존 이미지와 다른 태그를 붙인다 (같은 태그로 빌드하면 기존 이미지를 덮어쓴다).
FROM ros:noetic

ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu mirror hash mismatch 완화 (Docker Desktop/WSL 간헐적)
RUN printf 'Acquire::Retries "5";\nAcquire::http::Pipeline-Depth "0";\n' \
    > /etc/apt/apt.conf.d/99docker-apt-fix

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata locales && \
    ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    locale-gen en_US.UTF-8 ko_KR.UTF-8 && \
    update-locale LANG=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    git curl wget nano vim tmux sudo bash-completion \
    build-essential cmake libyaml-cpp-dev libpcap-dev libgeographic-dev \
    python3-pip python3-dev \
    python3-scipy python3-numpy \
    iputils-ping net-tools mesa-utils \
    ros-noetic-rosbridge-server \
    ros-noetic-rosbridge-suite \
    ros-noetic-tf2-sensor-msgs \
    ros-noetic-cv-bridge \
    dmz-cursor-theme xcursor-themes \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lbc.txt requirements-mgeo.txt requirements-grpc.txt requirements-linknode.txt /tmp/
RUN pip3 install --no-cache-dir --upgrade "pip<24.1"

# 시스템 Python 3.8 용 CUDA PyTorch. src/traffic_light/live_infer.py 가 이
# 인터프리터를 사용한다. 기본 이미지는 RTX 40 시리즈용이며 신호등 ResNet18과
# 센서퓨전 YOLO가 모두 이 CUDA 11.8 환경을 사용한다. RTX 50 전용 Python/CUDA
# 환경은 이 파일이 아니라 Dockerfile.rtx50에서 추가한다.
RUN pip3 install --no-cache-dir \
      torch==2.1.2 \
      torchvision==0.16.2 \
      --index-url https://download.pytorch.org/whl/cu118

# Ubuntu/ROS가 distutils로 설치한 Python 패키지는 pip가 제거할 수 없으므로,
# 추론 의존성에 필요한 버전을 /usr/local에 overlay한다.
RUN pip3 install --no-cache-dir --ignore-installed \
      "PyYAML>=5.4" \
      "numpy==1.24.4" \
      "scipy==1.10.1" \
      "matplotlib==3.7.5" \
      "pillow==10.4.0" \
      "psutil>=5.8" \
      "protobuf==3.20.3"

# RTX 40 기본 이미지에서 검증한 Ultralytics 버전을 고정해 재빌드 날짜에
# 따른 YOLO 동작/의존성 변경을 막는다. RTX 50은 별도 requirements를 사용한다.
RUN pip3 install --no-cache-dir -r /tmp/requirements-lbc.txt && \
    pip3 install --no-cache-dir -r /tmp/requirements-mgeo.txt && \
    pip3 install --no-cache-dir -r /tmp/requirements-grpc.txt && \
    pip3 install --no-cache-dir -r /tmp/requirements-linknode.txt && \
    pip3 install --no-cache-dir tqdm ultralytics==8.4.150 && \
    rm -f /tmp/requirements-*.txt && \
    python3 -c "from scipy.spatial import cKDTree; import cv2; import numpy; import h5py; import pyproj"

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "[ -f /root/ws/devel/setup.bash ] && source /root/ws/devel/setup.bash" >> /root/.bashrc && \
    echo "export LANG=en_US.UTF-8" >> /root/.bashrc && \
    echo "export LC_ALL=en_US.UTF-8" >> /root/.bashrc

# Qt expects XDG_RUNTIME_DIR to exist, be owned by the current user and have
# mode 0700. Do not inherit the host WSL path: it is commonly absent inside
# the container. The helper also isolates per-run Qt geometry so a saved
# dual-monitor position cannot make rqt/RViz open off-screen.
COPY ros-gui /usr/local/bin/ros-gui
RUN chmod 0755 /usr/local/bin/ros-gui && \
    install -d -m 0700 /tmp/ros-xdg-runtime

WORKDIR /root/ws

ENV QT_X11_NO_MITSHM=1
ENV XDG_RUNTIME_DIR=/tmp/ros-xdg-runtime
ENV XCURSOR_PATH=/usr/share/icons
ENV XCURSOR_THEME=DMZ-White

CMD ["bash"]
