#!/usr/bin/env bash
# Fixed-model RGB shadow. All state/output stays outside the team checkout.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile_override="${ML_GPU_PROFILE:-}";root_override="${ML_SHADOW_ROOT:-}"
config="${ML_LOCAL_ENV:-$repo/config/local.env}"
if [[ -f "$config" ]]; then set -a; source "$config"; set +a; fi
source "$repo/scripts/gpu_profiles.sh"
[[ -z "$profile_override" ]] || ML_GPU_PROFILE="$profile_override"
[[ -z "$root_override" ]] || ML_SHADOW_ROOT="$root_override"
root="${ML_SHADOW_ROOT:-${ML_OUTPUT_ROOT:-$repo/../outputs}/shadow}"
command="${1:-help}";if [[ $# -gt 0 ]]; then shift; fi
usage() {
  cat <<'TXT'
./scripts/shadow.sh demo                       # included image + models
./scripts/shadow.sh start                      # ROS, 10 minutes, 1 RGB/s
./scripts/shadow.sh start --seconds 600 --hz 1  # optional limits
./scripts/shadow.sh start --transport udp      # Windows UDP fallback
./scripts/shadow.sh status
./scripts/shadow.sh stop                       # flush last review window
./scripts/shadow.sh replay /path/to/session    # all original RGB offline
./scripts/shadow.sh bridge                     # start ROS bridge before GUI setup

One-time: set ML_SHADOW_ROOT in config/local.env to your output drive.
Set ML_GPU_PROFILE=rtx40/rtx50 (or auto); ML_SHADOW_DEVICE=cpu for a CPU check.
Front camera: 1280x720, unchanged fixed extrinsics, Capture Mode OFF.
ROS: 127.0.0.1:9090, /image_jpeg/compressed, Camera-1. Click Connect once.
UDP: destination Windows 127.0.0.1:9291, source9290; Windows py.exe required.
No driving or automatic training admission. stop does not alter vehicle state.
TXT
}
ensure_bridge() {
  local name="${ML_SHADOW_ROS_CONTAINER:-ml-shadow-ros}" image="${ML_SHADOW_ROS_IMAGE:-ml-shadow-ros:noetic-20261129}"
  if docker inspect "$name" >/dev/null 2>&1; then
    if [[ "$(docker inspect -f '{{index .Config.Labels "org.ml-shadow.owner"}}' "$name")" != team-repo ]]; then
      echo "[ERROR] $name is not managed by this script; choose ML_SHADOW_ROS_CONTAINER." >&2;return 1
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "$name")" != true ]]; then docker start "$name" >/dev/null; fi
  else
    if ! docker image inspect "$image" >/dev/null 2>&1; then docker build -t "$image" "$repo/docker/shadow-ros"; fi
    docker run -d --name "$name" --label org.ml-shadow.owner=team-repo -p "${ML_SHADOW_ROS_PORT:-9090}":9090 \
      -v "$repo":/workspace:ro -v "$root":/shadow "$image" >/dev/null
  fi
  bridge="$name"
}
resolve_runtime() {
  ml_resolve_gpu_profile "${ML_GPU_PROFILE:-auto}";device="${ML_SHADOW_DEVICE:-cuda}";gpu=()
  if [[ "$device" == cuda ]]; then gpu=(--gpus all); elif [[ "$device" != cpu ]]; then echo '[ERROR] device must be cpu or cuda' >&2;return 1; fi
}
run_worker() {
  local archive="$1" output="$2";shift 2
  docker run --rm "${gpu[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH=/workspace/src -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 \
    -v "$repo":/workspace:ro -v "$archive":/archive:ro -v "$output":/result -w /workspace \
    "$ML_RESOLVED_IMAGE" python3 -m quality_estimation.shadow_worker --archive /archive --output /result/review \
    --perception-release models/perception/m-v6-r2-20261128 --bundle models/review/m-v6-r2-20261128/primary \
    --gp-bundle models/review/m-v6-r2-20261128/GP_only --rf-bundle models/review/m-v6-r2-20261128/RF_only --device "$device" "$@"
}
case "$command" in
help|-h|--help) usage ;;
bridge)
  mkdir -p "$root";root="$(realpath "$root")";ensure_bridge;echo "ROS bridge: $bridge, port ${ML_SHADOW_ROS_PORT:-9090}" ;;
demo)
  resolve_runtime;mkdir -p "$root";out="$root/demo-$(date -u +%Y%m%dT%H%M%S)-$$";mkdir "$out"
  run_worker "$repo/models/examples" "$out";echo "Output: $out" ;;
replay)
  [[ $# == 1 ]] || { usage;exit 2; };resolve_runtime
  source_session="$(realpath "$1")";archive="$source_session/archive";[[ -f "$source_session/rgb.jsonl" ]] && archive="$source_session"
  [[ -f "$archive/rgb.jsonl" ]] || { echo '[ERROR] RGB archive missing' >&2;exit 1; }
  mkdir -p "$root";out="$root/replay-$(date -u +%Y%m%dT%H%M%S)-$$";mkdir "$out";run_worker "$archive" "$out";echo "Output: $out" ;;
status|stop)
  [[ -f "$root/active.json" ]] || { echo 'No recorded shadow session.';exit 0; }
  mapfile -t state < <(python3 - "$root/active.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]));print(s['session']);print(s['worker_container']);print(s['transport'])
PY
)
  session="${state[0]}";worker="${state[1]}"
  if [[ "$command" == stop ]]; then
    if [[ -d "$session/archive" ]]; then touch "$session/archive/STOP"; fi
    # A failed/missing receiver must not leave an idle GPU worker behind.
    for ((i=0;i<40;i++)); do
      [[ -f "$session/archive/receiver_done.json" ]] && break
      [[ "$(docker inspect -f '{{.State.Running}}' "$worker" 2>/dev/null || true)" != true ]] && break
      sleep .25
    done
    if [[ -d "$session/archive" && ! -f "$session/archive/receiver_done.json" ]]; then touch "$session/archive/STOP_WORKER"; fi
    for ((i=0;i<80;i++)); do
      [[ "$(docker inspect -f '{{.State.Running}}' "$worker" 2>/dev/null || true)" != true ]] && break
      sleep .25
    done
  fi
  python3 - "$session" <<'PY'
from pathlib import Path
import json,sys
p=Path(sys.argv[1]);print('Session:',p)
for name in ['archive/receiver_status.json','review/status.json']:
 q=p/name
 if q.exists():print(name,json.dumps(json.loads(q.read_text()),ensure_ascii=False))
PY
  docker inspect -f 'Worker running: {{.State.Running}}; exit: {{.State.ExitCode}}' "$worker" 2>/dev/null || true ;;
start)
  transport=ros;seconds=600;hz=1;window=100
  while [[ $# -gt 0 ]]; do
    case "$1" in --transport) transport="$2";shift 2;;--seconds)seconds="$2";shift 2;;--hz)hz="$2";shift 2;;--window)window="$2";shift 2;;*) echo "Unknown option: $1" >&2;exit 2;;esac
  done
  python3 - "$seconds" "$hz" "$window" <<'PY'
import sys
s,h,w=map(float,sys.argv[1:]);assert 0<s<=14400 and 0<h<=20 and 1<=w<=1000 and w==int(w),'Invalid bounded duration/rate/window'
PY
  [[ "$transport" == ros || "$transport" == udp ]] || { echo 'transport must be ros or udp' >&2;exit 2; }
  mkdir -p "$root";root="$(realpath "$root")"
  if [[ -f "$root/active.json" ]]; then
    old="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["worker_container"])' "$root/active.json")"
    [[ "$(docker inspect -f '{{.State.Running}}' "$old" 2>/dev/null || true)" != true ]] || { echo 'Shadow already running; use status/stop.' >&2;exit 1; }
  fi
  resolve_runtime
  if [[ "$transport" == ros ]]; then ensure_bridge; else command -v py.exe >/dev/null;[[ "$root" == /mnt/* ]] || { echo 'UDP output must be a Windows-mounted drive.' >&2;exit 1; };fi
  name="$(date -u +%Y%m%dT%H%M%S)-$$";session="$root/$name";mkdir "$session";worker="ml-shadow-worker-$name"
  # The worker initializes before the receiver starts, avoiding startup losses.
  maximum="$(python3 -c 'import sys;print(float(sys.argv[1])+120)' "$seconds")"
  docker run -d --name "$worker" --label org.ml-shadow.owner=team-repo "${gpu[@]}" --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/workspace/src -e OPENBLAS_NUM_THREADS=2 -e OMP_NUM_THREADS=2 \
    -v "$repo":/workspace:ro -v "$session":/session -w /workspace "$ML_RESOLVED_IMAGE" \
    python -m quality_estimation.shadow_worker --archive /session/archive --output /session/review --perception-release models/perception/m-v6-r2-20261128 \
    --bundle models/review/m-v6-r2-20261128/primary --gp-bundle models/review/m-v6-r2-20261128/GP_only --rf-bundle models/review/m-v6-r2-20261128/RF_only \
    --device "$device" --watch --maximum-seconds "$maximum" --window "$window" >/dev/null
  python3 - "$root/active.json" "$session" "$worker" "$transport" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(session=sys.argv[2],worker_container=sys.argv[3],transport=sys.argv[4]),indent=2)+'\n');t.replace(p)
PY
  ready=false
  for ((i=0;i<120;i++)); do
    if [[ -f "$session/review/READY.json" ]]; then ready=true;break;fi
    [[ "$(docker inspect -f '{{.State.Running}}' "$worker")" == true ]] || break
    sleep .25
  done
  if [[ "$ready" != true ]]; then docker logs "$worker" >&2;echo 'Worker did not become ready. Inspect status before restarting.' >&2;exit 1;fi
  if [[ "$transport" == ros ]]; then
    relative="${session#"$root"/}"
    nohup docker exec "$bridge" bash -lc 'source /opt/ros/noetic/setup.bash; export PYTHONPATH=/workspace/src:$PYTHONPATH; exec python3 -m quality_estimation.ros_rgb_archive "$@"' _ \
      --output "/shadow/$relative/archive" --seconds "$seconds" --capture-hz "$hz" >"$session/receiver.log" 2>&1 </dev/null &
  else
    cp "$repo/src/quality_estimation/camera_udp.py" "$session/camera_udp.py"
    nohup py.exe -3 "$(wslpath -w "$session/camera_udp.py")" --output "$(wslpath -w "$session/archive")" --seconds "$seconds" --capture-hz "$hz" \
      >"$session/receiver.log" 2>&1 </dev/null &
  fi
  echo "$!" >"$session/receiver_launcher.pid"
  for ((i=0;i<40;i++)); do [[ -f "$session/archive/archive.json" ]] && break;sleep .1;done
  echo "Started $transport shadow: $session";echo './scripts/shadow.sh status   |   ./scripts/shadow.sh stop' ;;
*)usage;exit 2;;
esac
