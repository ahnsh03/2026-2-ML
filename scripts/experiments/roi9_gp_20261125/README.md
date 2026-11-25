# Nine-region GP study, 2026-11-25/26

Current integration baseline: fixed Medium v6 perception, `roi9-front-v1`, 28 ordered inputs, 2,480 common RF/GP training rows. Main target is one whole-frame white/yellow lane IoU. Near/far image-band recalls are offline-supervised omission diagnostics, not online ROI IoUs. Spatial probabilities remain diagnostic-only because far-recall intervals under-cover on the new release.

See the [measured results and limitations](../../../docs/experiments/ROI9_GP와_26R1_검증_20261126.md), [public protocols](../../../docs/baselines/roi9_gp_20261126/protocols/), [feature recipe](../../../config/region_feature_recipe_v3.json), and [runtime guide](../../../docs/guides/개발환경.md). Raw media, complete row-level indices, model artifacts and logs stay on the experiment volume. The public protocols describe the actual run after completion, not a new preregistration. Original plans remain unchanged in the private study.

## Units and inputs

| Unit | Entry point | Required inputs / outputs |
|---|---|---|
| Fixed-camera stationary collection | `capture_stationary.py` | External simulator proto, planned poses/conditions, output mapping; preserves raw channels and verifies stationary before/after state. 26R1 file-backed acceptance does not validate live synchronous collection |
| Capture registration | `register_capture.py` | Accepted collections, protocol and existing index; checks expected rows/IDs/media and registers versioned new source index |
| GPU extraction | `extract.py` | Registered index, source root, fixed checkpoint; writes 372 candidate columns and separate GT counts/metadata plus file/schema/index hashes |
| Feature comparison | `study.py` | Packs, v2 starting recipe, grouped protocol, optional explicit column views; RF/GP screen and frozen candidate records |
| Quality-label acquisition | `fit_finalists.py` | Packs, selected columns, role protocol; retrospective 192-row acquisition/random/all624 controls and coupling trials |
| Broader quality fitting | `coverage.py` | Packs, coverage protocol, selected28; input-only2,496 selection, complete-label2,480 fitting, geographic folds and frozen RF/GP |
| Additive kernel / BO | `additive_search.py` | Same packs/roles/columns; equal-budget EI BO and random controls. This experiment did not improve the baseline |
| Archival geometry / scoring | `register_geometry.py`, `evaluate.py` | Original map/plans, packs, frozen models; record role/clearance before scores. Broader-training follow-up uses previously opened archive data as diagnostics |
| Fixed-budget comparison | `policy_study.py` | Packs, index, frozen run and completed evaluation; session-preserving64-frame windows and ceil10% budgets |
| Release calibration / scoring | `prepare26_calibration.py`, `score26.py` | Study directory, registered26R1 calibration/test packs, weights locked before scores; do not reuse opened test data for an independent claim |
| Bundle / publication | `export.py`, `package_bundle.py`, `publish.py` | Frozen weights/calibration/source packs; portable GP, exact private training membership, aggregate public evidence |
| Figures | `make_figures.py`, `make26_figures.py` | Exact RGB replay/softmax and matching source GT; preserve native aspect ratio and per-case receipts |

Commands expose their exact options through `--help`. Run CPU fitting with `PYTHONPATH=src:scripts` and the study dependency set. GPU extraction uses the rtx50 image; the rtx40 image remains a separate profile. Outputs use new directories; interrupted histories and failed collection records remain on disk.

## Refit and verify

Prepare the registered study packs (including logical geographic `zNN` partitions), schema and hash receipts before fitting. `STUDY_PACKS` is the handoff pack directory, not a raw image directory. `NEW_FIT` must be a new result directory.

```bash
PYTHONPATH=src:scripts python3 scripts/experiments/roi9_gp_20261125/coverage.py \
  --packs "$STUDY_PACKS" \
  --protocol docs/baselines/roi9_gp_20261126/protocols/coverage.json \
  --features docs/baselines/roi9_gp_20261126/protocols/selected_features.json \
  --output "$NEW_FIT"

PYTHONPATH=src python3 -m quality_estimation.spatial_bundle \
  --bundle "$FROZEN_BUNDLE" --simulator-release 26.R1
```

GPU-free GP inference uses NPZ rather than pickle. The RF joblib needs the training scikit-learn version. The bundle checker verifies hashes, feature/checkpoint/ROI context, complete training membership and disjoint role IDs; geographic independence must still be checked from coordinates/link provenance.

The RGB-only scorer, review queue and deferred GT join are reusable modules rather than study-specific paths:

```bash
PYTHONPATH=src python3 -m quality_estimation.infer_spatial_images --help
PYTHONPATH=src python3 -m quality_estimation.review_spatial --help
PYTHONPATH=src python3 -m quality_estimation.join_spatial_gt --help
```

A completed queue or GT join is not approval to retrain. Candidate admission remains false until role, visibility and operational-policy review. No new scientific perception retraining or live26R1 ROS acceptance is claimed by this study.
