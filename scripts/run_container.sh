#!/usr/bin/env bash
# 인지 모델 학습 이미지에서 이 레포의 코드를 실행한다 (추론·시각화·export).
# 인지 모듈과 TwinLiteNet+ 원본(서브모듈)은 이 레포 안에 있다. 데이터·체크포인트만 밖에서 마운트한다.
#
# 사용 예:
#   scripts/run_container.sh python3 -m visualization.visualize_inference --help
#
# 경로와 이미지 이름은 config/local.env(Git 제외)에서 읽는다.
# 처음 한 번 config/local.env.example을 복사해 자기 PC 경로로 채운다.
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
ML_OUTPUT_ROOT="${ML_OUTPUT_ROOT:-$ML_WS/outputs}"

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command...>" >&2
  echo "  e.g. $0 python3 -m visualization.visualize_inference --help" >&2
  exit 2
fi

missing=()
for name in PERCEPTION_DATA PERCEPTION_CKPT_ROOT PERCEPTION_IMAGE; do
  if [[ -z "${!name:-}" ]]; then
    missing+=("$name")
  fi
done
if ((${#missing[@]})); then
  echo "[ERROR] 설정이 없습니다: ${missing[*]}" >&2
  echo "        config/local.env.example을 config/local.env로 복사해 채우세요" >&2
  exit 1
fi
if [[ ! -f "$ML_REPO/third_party/TwinLiteNetPlus/model/model.py" ]]; then
  echo "[ERROR] TwinLiteNet+ 서브모듈이 없습니다: git submodule update --init third_party/TwinLiteNetPlus" >&2
  exit 1
fi
for path in "$PERCEPTION_DATA" "$PERCEPTION_CKPT_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] 경로가 없습니다: $path" >&2
    exit 1
  fi
done
mkdir -p "$ML_OUTPUT_ROOT"

echo "[DATA]   $PERCEPTION_DATA -> /data (ro)" >&2
echo "[CKPT]   $PERCEPTION_CKPT_ROOT -> /ckpt (ro)" >&2
echo "[OUTPUT] $ML_OUTPUT_ROOT -> /outputs (rw)" >&2

tty_flags=()
if [[ -t 0 && -t 1 ]]; then
  tty_flags=(-it)
fi

# 결과 파일이 root 소유가 되지 않도록 호스트 사용자로 실행한다.
# 이미지의 /root 는 다른 사용자가 들어갈 수 없으므로 마운트는 /root 밖에 둔다.
exec docker run --rm "${tty_flags[@]}" --gpus all --shm-size 8g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$ML_REPO":/ml:ro \
  -v "$ML_OUTPUT_ROOT":/outputs \
  -v "$PERCEPTION_DATA":/data:ro \
  -v "$PERCEPTION_CKPT_ROOT":/ckpt:ro \
  -e PERCEPTION_DATA=/data \
  -e PERCEPTION_CKPT_ROOT=/ckpt \
  -e ML_OUTPUT_ROOT=/outputs \
  -e PYTHONPATH=/ml/src \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -w /ml \
  "$PERCEPTION_IMAGE" "$@"
