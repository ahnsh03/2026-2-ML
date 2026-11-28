# Thin ROS transport; perception stays in the separate rtx40/rtx50 runtime.
FROM ros:noetic
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-rosbridge-server python3-opencv \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1
CMD ["roslaunch", "rosbridge_server", "rosbridge_websocket.launch", "address:=0.0.0.0", "port:=9090"]
