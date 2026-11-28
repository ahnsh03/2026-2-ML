# GP contrast and26R1 shadow — 2026-11-29 unit

Frozen reference:9-region front profile, r2 perception. **Both the25% z-rank blend and57-column GP-derived quality input failed confirmation and are excluded.** RF/GP remain parallel controls until larger selection/retraining comparisons. [Results](../../../docs/experiments/대비특징_검수와_shadow_20261129.md) · [Public evidence](../../../docs/baselines/shadow_cycle_20261129/README.md) · [Simple runtime commands](../../../docs/guides/shadow_실행.md).

| Script | Input / role |
|---|---|
| `compare_gp_contrast.py` | Parent fit, disjoint Meta Cal and Dev. Direct z/rank selection and74-column meta-input controls. No confirmation access. |
| `quality_meta_development.py` | Small disjoint-fit quality meta-input pilot; not the matched2500-row experiment. |
| `quality_oof_development.py` | Matched2500-row Q52/55/57 study, spatial3-fold25m-buffered OOF GP-derived training inputs; Cal reserved for residual envelope. |
| `register_confirmation.py` | Every slot in the completed534-row collection, raw hashes and restoration; no score-based filtering. |
| `extract_environment.py` | RGB-only frozen400-column descriptors; no GT/SDK weather features. |
| `assess_contrast_confirmation.py` | Frozen selector, plan, version-matched pack and environment. RF/GP/blend, random1000, six-block bootstrap, global/window100 populations. |
| `assess_quality_input.py` | Frozen GP57 quality candidate against GP52/RF52. No test tuning. |
| `qa_visibility.py` | Every native Semantic GT count versus scored pack, offline local-contrast cue and first-eligible unselected cases. |
| `export_qa_gallery.py` | Replays exact original CPU16-frame batches, asserts every case TP/FP/FN matches pack;16:9 component panels. |
| `drive_shadow.py` | Bounded6-route4km/h functional verification.26R1 UDP steering correction, no quality-based control, restore after last UDP packet. No paired GT. |
| `photometric_probe.py` | One parked pose,13→05→06→07→13h and restoration. Excluded from accuracy/training. Requires locally supplied simulator proto/SDK. |
| `assess_photometric_probe.py` | Early/late near/hood/top luma; no physical visibility label or policy tuning. |
| `summarize_shadow.py` | Immutable JPEG hashes, raw/inferred/skipped membership and receiver-to-score timing; excludes simulator exposure/network delay. |
| `verify_shadow_replay.py` | Same processed IDs, ordered features, exact windows/budgets/batches and core implementation/model hashes; numeric differences and selections. |
| `plot_development.py` / `plot_confirmation.py` | Static scientific figures with source/input/artifact hashes; development and confirmation separated. |

The raw registry, original captures, OOF folds and rejected model files stay on the experiment volume. Run the version-bound extractor `scripts/experiments/roi9_gp_20261125/extract.py` with `--perception-release models/perception/m-v6-r2-20261128`. Its receipt must match the registry and r2 identity. The534-row quality hypothesis was frozen during capture but before any test feature/target/prediction access; the selector and collection plan were frozen before capture.

Use `--help` for explicit paths. Example verification, from the repository:

```bash
./scripts/shadow.sh replay "$SHADOW_SESSION" --processed-only
PYTHONPATH=src python3 scripts/experiments/shadow_cycle_20261129/verify_shadow_replay.py \
  --live "$SHADOW_SESSION" --replay "$SHADOW_REPLAY" --output "$NEW_RECEIPT"
```

Processed-only replay preserves the recorded inference batch schedule; generic replay may change GPU batching and slightly perturb RF splits. The historical293-frame worker predates this batch-control addition; its comparison explicitly records a worker-source change while requiring identical core scorer hashes. The newer146-frame run uses identical worker source for live and replay.

Static534 rows are six physical scene blocks, not534 independent roads. Earlier590 frames are already-opened history. Native support errors/low RGB contrast do not establish physical visibility or safety. Controlled ROS/UDP protocol replays and the GT-free driving/photometric sessions are not independent accuracy/training datasets. Actual40-series CUDA hardware remains untested.
