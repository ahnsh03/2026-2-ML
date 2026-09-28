#!/usr/bin/env bash
# 데이터 수집용 ROS Noetic 컨테이너를 이 레포 전용으로 띄우고, 그 안에서 명령을 실행한다.
#
#   scripts/run_ros.sh up            컨테이너를 만들고(없으면) 시작한다
#   scripts/run_ros.sh build         ros/ 워크스페이스를 catkin_make 한다
#   scripts/run_ros.sh shell         ROS 환경이 잡힌 bash
#   scripts/run_ros.sh exec <명령…>  ROS 환경에서 명령 하나를 실행한다
#   scripts/run_ros.sh status | down
#
# 이미지는 config/local.env 의 ROS_IMAGE 를 그대로 쓴다 (docker/ros-noetic/Dockerfile 로 만든 이미지).
# 이 레포(/ml)와 ML 데이터 루트(/data)만 마운트한다. 다른 프로젝트 폴더는 마운트하지 않는다.
# 자세한 설명은 docs/05_구현계획/개발환경.md 를 본다.
set -euo pipefail

ML_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ML_LOCAL_ENV:-$ML_REPO/config/local.env}"
if [[ -f "$CONFIG" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG"
  set +a
fi
ML_WS="${ML_WS:-$(dirname "$ML_REPO")}"
ML_DATA_ROOT="${ML_DATA_ROOT:-$ML_WS/data}"
ROS_CONTAINER="${ROS_CONTAINER:-ml-ros-noetic}"
LABEL="ml2026.repo"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

# 이 스크립트가 만든 컨테이너만 다룬다. 같은 이름의 다른 컨테이너가 있으면 멈춘다.
container_state() {
  local owner
  if ! owner="$(docker inspect -f "{{ index .Config.Labels \"$LABEL\" }}" "$ROS_CONTAINER" 2>/dev/null)"; then
    echo "absent"
    return
  fi
  [[ "$owner" == "$ML_REPO" ]] || die "컨테이너 $ROS_CONTAINER 는 이 레포가 만든 것이 아닙니다 (label=$owner). ROS_CONTAINER 를 다른 이름으로 정하세요"
  docker inspect -f '{{.State.Running}}' "$ROS_CONTAINER"
}

require_running() {
  [[ "$(container_state)" == "true" ]] || die "$ROS_CONTAINER 가 실행 중이 아닙니다: $0 up"
}

# catkin setup 스크립트는 호출한 쪽의 위치 인자를 자기 옵션으로 읽는다. 인자를 비운 뒤 source 하고 되돌린다.
ROS_ENV='ros_args=("$@"); set --; source /opt/ros/noetic/setup.bash; if [[ -f /ml/ros/devel/setup.bash ]]; then source /ml/ros/devel/setup.bash; fi; set -- "${ros_args[@]}"'

cmd="${1:-}"
[[ -n "$cmd" ]] || die "usage: $0 {up|build|shell|exec <명령…>|status|down}"
shift

case "$cmd" in
  up)
    [[ -n "${ROS_IMAGE:-}" ]] || die "ROS_IMAGE 가 없습니다. config/local.env.example 을 참고해 채우세요"
    state="$(container_state)"
    if [[ "$state" == "absent" ]]; then
      mkdir -p "$ML_DATA_ROOT"
      proto_args=()
      if [[ -n "${MORAI_GRPC_PROTO_ROOT:-}" ]]; then
        [[ -d "$MORAI_GRPC_PROTO_ROOT" ]] || die "MORAI_GRPC_PROTO_ROOT 경로가 없습니다: $MORAI_GRPC_PROTO_ROOT"
        proto_args=(-v "$MORAI_GRPC_PROTO_ROOT":/opt/morai_grpc_proto:ro -e MORAI_GRPC_PROTO_ROOT=/opt/morai_grpc_proto)
      fi
      echo "[REPO] $ML_REPO -> /ml (rw, catkin build/devel 은 ros/ 아래에 생긴다)" >&2
      echo "[DATA] $ML_DATA_ROOT -> /data (rw)" >&2
      # MORAI UDP·gRPC 가 127.0.0.1 로 붙으므로 host network 를 쓴다.
      # 결과 파일이 root 소유가 되지 않도록 호스트 사용자로 실행한다 (이미지의 /root 는 쓰지 않는다).
      docker run -d --name "$ROS_CONTAINER" --label "$LABEL=$ML_REPO" \
        --network host --shm-size 2g \
        --user "$(id -u):$(id -g)" \
        -e HOME=/tmp -e ROS_HOME=/tmp/.ros \
        -e ML_DATA_ROOT=/data \
        -e ROS_MASTER_URI=http://127.0.0.1:11311 \
        -e PYTHONDONTWRITEBYTECODE=1 \
        "${proto_args[@]}" \
        -v "$ML_REPO":/ml \
        -v "$ML_DATA_ROOT":/data \
        -v /etc/localtime:/etc/localtime:ro \
        -w /ml \
        "$ROS_IMAGE" tail -f /dev/null >/dev/null
    elif [[ "$state" == "false" ]]; then
      docker start "$ROS_CONTAINER" >/dev/null
    fi
    echo "[OK] $ROS_CONTAINER 실행 중 (image: $(docker inspect -f '{{.Config.Image}}' "$ROS_CONTAINER"))" >&2
    # host network 라서 다른 ROS 컨테이너의 roscore 와 포트가 겹친다.
    if command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ':11311 '; then
      echo "[WARN] 호스트 11311 포트에 이미 ROS master 가 있습니다. 수집 노드가 그 master 에 붙습니다." >&2
      echo "       다른 ROS 컨테이너를 쓰지 않을 때 수집하세요." >&2
    fi
    ;;
  build)
    require_running
    docker exec "$ROS_CONTAINER" bash -c "$ROS_ENV; catkin_make -C /ml/ros $*"
    ;;
  shell)
    require_running
    docker exec -it "$ROS_CONTAINER" bash -c "$ROS_ENV; exec bash -i"
    ;;
  exec)
    require_running
    [[ $# -gt 0 ]] || die "usage: $0 exec <명령…>"
    tty_flags=()
    if [[ -t 0 && -t 1 ]]; then
      tty_flags=(-it)
    fi
    docker exec "${tty_flags[@]}" "$ROS_CONTAINER" bash -c "$ROS_ENV; exec \"\$@\"" _ "$@"
    ;;
  status)
    state="$(container_state)"
    echo "$ROS_CONTAINER: $state"
    ;;
  down)
    state="$(container_state)"
    if [[ "$state" != "absent" ]]; then
      docker rm -f "$ROS_CONTAINER" >/dev/null
      echo "[OK] $ROS_CONTAINER 삭제" >&2
    fi
    ;;
  *)
    die "usage: $0 {up|build|shell|exec <명령…>|status|down}"
    ;;
esac
