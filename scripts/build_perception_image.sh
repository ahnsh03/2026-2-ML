#!/usr/bin/env bash
set -euo pipefail
ML_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
  rtx40) profile=rtx40; tag="${2:-ml-perception:rtx40-cu118}" ;;
  rtx50) profile=rtx50; tag="${2:-ml-perception:rtx50-cu128}" ;;
  *) echo "usage: $0 {rtx40|rtx50} [new-image-tag]" >&2; exit 2 ;;
esac
exec docker build -f "$ML_REPO/docker/perception/Dockerfile.$profile" -t "$tag" "$ML_REPO"
