#!/usr/bin/env bash
# Source after local.env. An explicit profile is reproducible without a local GPU.
ml_resolve_gpu_profile() {
  local requested="${1:-auto}" capability
  if [[ "$requested" == auto ]]; then
    if ! capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1)"; then
      capability=""
    fi
    case "$capability" in
      12.*) requested=rtx50 ;;
      8.9) requested=rtx40 ;;
      *) echo "[ERROR] GPU 자동 판별 실패. ML_GPU_PROFILE=rtx40 또는 rtx50을 지정하세요" >&2; return 1 ;;
    esac
  fi
  case "$requested" in
    rtx40) ML_RESOLVED_IMAGE="${PERCEPTION_IMAGE_RTX40:-${PERCEPTION_IMAGE:-ml-perception:rtx40-cu118}}" ;;
    rtx50) ML_RESOLVED_IMAGE="${PERCEPTION_IMAGE_RTX50:-ml-perception:rtx50-cu128}" ;;
    *) echo "[ERROR] 지원하지 않는 GPU 프로필: $requested" >&2; return 1 ;;
  esac
  ML_RESOLVED_PROFILE="$requested"
}
