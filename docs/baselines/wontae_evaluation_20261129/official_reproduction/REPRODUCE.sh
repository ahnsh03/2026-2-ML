#!/usr/bin/env bash
# bash REPRODUCE.sh /path/to/preserved-official-input-root /path/to/NEW-output
# Replays frozen inputs/models; never starts ROS or modifies NAS.
set -euo pipefail
official_inputs="$(realpath "$1")"
official_output="$2"
[[ ! -e "$official_output" ]] || { echo 'Use a new output directory.' >&2; exit 2; }
mkdir -p "$official_output"
official_output="$(realpath "$official_output")"
official_release="$official_inputs/release/ml_scene_release_r2"
image=ml-perception:rtx50-cu128
python3 "$official_release/code/scripts/verify_environment_release.py" --root "$official_release" > "$official_output/release_verification.json"

replay() {
  local name="$1" manifest="$2" data="$3" budget="$4" bundle="$5"
  mkdir "$official_output/$name"
  docker run --rm --gpus all --shm-size 1g --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e PYTHONPATH=/release/code/src:/release/code/ros/src/data_collection/src \
    -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
    -v "$official_release":/release:ro -v "$data":/data:ro \
    -v "$official_inputs/reference_models/RF_only":/rf_bundle:ro \
    -v "$official_output/$name":/output -w /release/code "$image" \
    python3 -B -m quality_estimation.infer_versioned_review --bundle "$bundle" \
    --adaptation-run /release/models/perception --manifest "$manifest" --data-root /data \
    --output /output/gpu --device cuda --batch-size 4 --budget "$budget" \
    > "$official_output/$name.log" 2>&1
}
replay standalone48 /release/examples/manifest.jsonl "$official_release/examples/rgb" 24 /release/models/review
replay replay590_primary /data/rgb_only.jsonl "$official_inputs/data_rgb" 59 /release/models/review
replay replay590_rf /data/rgb_only.jsonl "$official_inputs/data_rgb" 59 /rf_bundle

python3 - "$official_output" "$official_inputs" <<'PY'
from pathlib import Path
import hashlib,json,sys
out,inputs=map(Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
freeze=dict(primary_candidates_sha256=sha(out/'replay590_primary/gpu/candidates.jsonl'),
            primary_run_sha256=sha(out/'replay590_primary/gpu/run.json'),
            rf_candidates_sha256=sha(out/'replay590_rf/gpu/candidates.jsonl'),
            rf_run_sha256=sha(out/'replay590_rf/gpu/run.json'),
            source_index_sha256=sha(inputs/'nas_metadata/confirmation/registry/index_frozen.json'),
            phase='Both RGB-only selections frozen before Semantic evaluation',uses_ground_truth=False)
(out/'rgb_selection_frozen.json').write_text(json.dumps(freeze,indent=2)+'\n')
PY

# GT is mounted only after the RGB selections have been frozen.
mkdir "$official_output/gt590_recomputed"
docker run --rm --gpus all --shm-size 1g --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/release/code/src:/release/code/ros/src/data_collection/src \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
  -v "$official_release":/release:ro -v "$official_inputs/report/source":/tools:ro \
  -v "$official_inputs/data_rgb":/rgb:ro -v "$official_inputs/data_gt":/gt:ro \
  -v "$official_inputs/nas_metadata":/metadata:ro \
  -v "$official_output/replay590_primary/gpu":/queue:ro \
  -v "$official_output/rgb_selection_frozen.json":/freeze.json:ro \
  -v "$official_output/gt590_recomputed":/output -w /release/code "$image" \
  python3 /tools/recompute_r2_confirmation_gt.py --release /release --rgb-root /rgb --gt-root /gt \
  --source-index /metadata/confirmation/registry/index_frozen.json --queue /queue --freeze /freeze.json \
  --reference-pack /metadata/confirmation/packs_selected/confirmation.npz \
  --output /output/results --device cuda --batch-size 4 > "$official_output/recompute_gt590.log" 2>&1

echo "Completed RGB replays and raw GT recomputation: $official_output"
echo 'Compare with nas_metadata using source/compare_official_review_replay.py; differences must be reported, not tuned away.'
