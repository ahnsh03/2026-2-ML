# Full-replay environment and data-selection study

The current unit audits the original mixture, tests RGB environment discovery against simulator tags, compares same-v6-start selection policies, and rebuilds quality estimation after a perception update. Raw data and weights stay outside Git. The preceding subset-replay study remains under `selection_cycle_20261126/`.

## Frozen comparison

- Original train: **26,001** images, supported by the checkpoint's embedded train-manifest digest. Use the complete original pool, including all three camera views. The old4,096 subset used equal run/view quotas and changed original proportions.
- New admission bank:3,072 unique frames per method. Each step uses2new+14original;24,000updates, fixed v6 EMA start, fixed BN moments, BF16 forward/FP32 loss, identical learning rate and teacher regularization.
- Unique policies: Random, simulator-tag balanced random, RF calibrated global risk, GP calibrated global risk. Repeat Random/RF/GP at two optimizer seeds; balanced extraction has one diagnostic seed. Add one full-original-only continuation control, keeping eight long runs total. Simulator/RGB environment-threshold searches currently select exactly the GP global ordered bank; those policies alias its endpoint, rather than becoming fake independent trials.
- Symmetric parameter interpolation at alpha0.25/0.5/0.75/1.0 is evaluated on Dev/retention only. Compare methods at equal alpha separately from the final chosen pipeline.
- Quality-fit2500 includes834 previously opened diagnostic rows, explicitly retiring their test status. Dev937 and Cal612 retain their frozen roles. New geometry-selected26R1 confirmation is reserved from training and all tuning. Zero-union IoU remains undefined.

## Commands and artifacts

Set `STUDY_ROOT`, `PARENT_STUDY`, `REPO_ROOT`, `CHECKPOINT_ROOT` to your local registered assets. Private source manifests retain original paths/hashes; do not commit them. These orchestrators use the existing WSL D/E mount convention and separate40/50 container profiles.

1. `cache_full_original.py`: validate original RGB/Semantic and compare **every cached two-head target and valid mask** with the frozen original version. Shared content-addressed cache, no raw copying.
2. `freeze_data.py`, `environment_features.py`: frozen quality roles and native RGB histograms. Preserve the source index and metadata, not only precomputed X/y.
3. `environment_study.py`: geographic CV of supervised RGB weather/lighting classification; official Faiss e-means on raw and learned representations, three clustering seeds, Lloyd/GMM controls. Learned probabilities plus e-means are explicitly **supervised representation learning**, not label-free weather discovery.
4. `quality_screen.py`: paired GP/RF Q regression, geographically buffered CV and feature ablations. Inputs and regression labels are version-bound.
5. `policy_study.py`: Cal-only relative references, Dev-only BO16 vs matched random16 at three review budgets; exact simulator tags and RGB clusters. `context_audit.py` exposes normality-label support and context-only regression controls.
6. `orchestrate.py`: runs/resumes frozen training queue. `development_queue.py` + `assess_development.py` evaluate symmetric coefficients, schedule all repeated unique banks, and freeze f1 without fresh confirmation.
7. `quality_queue.py`: regenerate X/y with f1, then refit GP/RF. `refit_policy.py` rebuilds f1 residual calibration/references and compares RGB grouping against a simulator oracle. Freeze all of this before test opening.
8. `collect_queue.py`, `register_confirmation.py`: parked-to-parked short26R1 drives, complete four-channel hashes, geometry-only thinning and source-role preservation. The optional local RPC-reset helper is private Windows-layout-specific tooling, passed explicitly as `--reset-script`.
9. `final_evidence_queue.py`: fresh alignment check, f0/f1 extraction, two-head evaluation, actual RGB/no-GT replay, GT reattachment,40CPU/50GPU compatibility, and acceptance without test retuning.
10. `figures.py`: standalone PNG/SVG aggregates; fresh figures are produced only after the frozen assessment exists.

Each stage preserves manifests, hashes, exact perception identity, model state type, source code digests and raw-data references. `_SUCCESS` is not a substitute for reading the corresponding report. A complete offline research run can reject the proposed update; that does not authorize live connection or automatic training admission.

## Interpretation

Weather-tag accuracy, clustering stability, IoU MAE, calibrated review retrieval and downstream perception gain answer different questions. The current moving Dev set confounds near-deficit with night; it cannot prove rejection of normal night scenes. Supported near precision/recall is a diagnostic proxy rather than visibility-aware operating truth. Unsupported relative references fall back to the common0.7 benchmark; no environmental normality is fabricated.
