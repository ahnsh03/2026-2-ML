# Full-replay environment and data-selection study

The current unit audits the original mixture, tests RGB environment discovery against simulator tags, compares same-v6-start selection policies, and rebuilds quality estimation after a perception update. Raw data and weights stay outside Git. The preceding subset-replay study remains under `selection_cycle_20261126/`.

**Completed, update not adopted.** Eight neural runs, version-matched quality/policy refitting, the independent618-frame cohort and standalone40-profile CPU/actual50GPU replay are complete. Neural and quality-retention gates passed, but review retrieval failed.59/62 review slots went to near-unsupported scenes and0/114 supported near-deficit frames were selected. See the [result report](../../../docs/experiments/환경검증_전체원본재생_20261128.md) and [public tables/16 figures](../../../docs/baselines/environment_cycle_20261128/README.md). These commands reproduce the frozen research cycle; they do not authorize online deployment or retuning on the opened cohort.

## Frozen comparison

- Original train: **26,001** images, supported by the checkpoint's embedded train-manifest digest. Use the complete original pool, including all three camera views. The old4,096 subset used equal run/view quotas and changed original proportions.
- New admission bank:3,072 unique frames per method. Each step uses2new+14original;24,000updates, fixed v6 EMA start, fixed BN moments, BF16 forward/FP32 loss, identical learning rate and teacher regularization.
- Unique policies: Random, simulator-tag balanced random, RF calibrated global risk, GP calibrated global risk. After all four first-stage Dev results, repeat Random and the two strongest eligible non-random banks, allowing balanced extraction to compete; the remaining bank has one diagnostic seed. Add one full-original-only continuation control, keeping eight long runs total. The amendment was logged after Random/balanced first-stage results but before RF/GP Dev or any fresh opening. Simulator/RGB environment-threshold searches currently select exactly the GP global ordered bank; those policies alias its endpoint, rather than becoming fake independent trials.
- Symmetric parameter interpolation at alpha0.25/0.5/0.75/1.0 is evaluated on Dev/retention only. Compare methods at equal alpha separately from the final chosen pipeline.
- Quality-fit2500 includes834 previously opened diagnostic rows, explicitly retiring their test status. Dev937 and Cal612 retain their frozen roles. New geometry-selected26R1 confirmation is reserved from training and all tuning. Zero-union IoU remains undefined.

## Commands and artifacts

Set `STUDY_ROOT`, `PARENT_STUDY`, `REPO_ROOT`, `CHECKPOINT_ROOT` to your local registered assets. Private source manifests retain original paths/hashes; do not commit them. These orchestrators use the existing WSL D/E mount convention and separate40/50 container profiles.

1. `cache_full_original.py`: validate original RGB/Semantic and compare **every cached two-head target and valid mask** with the frozen original version. Shared content-addressed cache, no raw copying. `audit_replay_exposure.py` reconstructs each frozen sampler and checks completed-run draw receipts.
2. `freeze_data.py`, `environment_features.py`: frozen quality roles and native RGB histograms. Preserve the source index and metadata, not only precomputed X/y.
3. `environment_study.py`: geographic CV of supervised RGB weather/lighting classification; [official Faiss e-means](https://github.com/facebookresearch/faiss/blob/main/contrib/e_means/README.md) on raw and learned representations, three clustering seeds, Lloyd/GMM controls. Learned probabilities plus e-means are explicitly **supervised representation learning**, not label-free weather discovery.
4. `quality_screen.py`: paired GP/RF Q regression, geographically buffered CV and feature ablations. Inputs and regression labels are version-bound.
5. `policy_study.py`: Cal-only relative references, Dev-only BO16 vs matched random16 at three review budgets; exact simulator tags and RGB clusters. `context_audit.py` exposes normality-label support and context-only regression controls.
6. `orchestrate.py`: runs/resumes frozen training queue. `development_queue.py` + `assess_development.py` evaluate symmetric coefficients, schedule all repeated unique banks, and freeze f1 without fresh confirmation.
7. `quality_queue.py`: regenerate X/y with f1, then refit GP/RF. `diagnose_uncertainty.py` separates point error, posterior scale and interval coverage. `fit_interval_envelope.py` compares two empirical interval recipes with a width-penalizing interval score. `diagnose_review_ties.py` measures finite-CDF saturation and compares identifier/standardized-margin ordering only within exact ties. `refit_policy.py` rebuilds f1 references and compares RGB grouping against a simulator oracle using the Dev-selected tie recipe. Freeze all of this before test opening. Existing f0 banks and calibration are not replaced by diagnostic candidates.
8. `collect_queue.py`, `register_confirmation.py`: parked-to-parked short26R1 drives, complete four-channel hashes, geometry-only thinning and source-role preservation. The optional local RPC-reset helper is private Windows-layout-specific tooling, passed explicitly as `--reset-script`.
9. `final_evidence_queue.py`: freeze training-only environment-mean controls (`context_controls.py`) to separate weather/time identity from per-image quality; fresh alignment check, f0/f1 extraction, two-head evaluation, actual RGB/no-GT replay, GT reattachment,40CPU/50GPU compatibility, and acceptance without test retuning.
10. `artifacts_queue.py`: after actual numeric assessment, export verified RGB/GT cases, standalone PNG/SVG figures and scrubbed public aggregates. No Git commit/push is performed by the worker.
11. `audit_confirmation_support.py`: post-assessment audit of sparse marking support, frozen selection allocation and per-road GP/RF errors. Reproduces the saved selection without GT, then attaches support diagnostics. It does not refit models, change thresholds or alter the original acceptance. Its negative evidence motivates a future, separately registered development/confirmation protocol.
12. `audit_policy_search.py`: audit all frozen Dev BO/random trials and propose the best eligible evaluated-union candidate as **development analysis only**. Simulator oracles cannot become runtime candidates, unequal/corrupt histories are rejected, existing files cannot be overwritten, and no confirmation data is opened. This tool does not build a deployable bundle or replace the rejected pair.

```bash
python3 scripts/experiments/environment_cycle_20261127/audit_policy_search.py \
  --policy "$FROZEN_POLICY" --output "$NEW_DEV_SEARCH_AUDIT"
```

Quality X/y extraction can run in FP32 on CPU while the frozen full-original-only neural control uses the GPU. The resumed extractor verifies exact checkpoint/index/pack hashes. The final worker explicitly waits for the completed control, then compares48 identical Dev IoU targets against actual50GPU extraction (no undefined-status changes; maximum absolute difference<=0.005), as well as RGB-only40CPU/50GPU replay and GT-role preservation. CPU preparation does not open fresh inputs or change any selection/adoption criterion. `check_runtime.py --prepare-cpu-only` can prepare its48-row CPU RGB replay before the control finishes, without marking compatibility/GT verification complete. Cached replays must match the exact bundle, input and perception identity, including artifact hashes.

`final_evidence_queue.py` checks the selected neural/quality development gates before opening fresh data. A rejected Dev candidate stays unopened for analysis and correction on development inputs. The final independent cohort is never used for this correction.

Each stage preserves manifests, hashes, exact perception identity, model state type, source code digests and raw-data references. `_SUCCESS` is not a substitute for reading the corresponding report. A complete offline research run can reject the proposed update; that does not authorize live connection or automatic training admission.

## Interpretation

Weather-tag accuracy, clustering stability, IoU MAE, calibrated review retrieval and downstream perception gain answer different questions. The current moving Dev set confounds near-deficit with night; it cannot prove rejection of normal night scenes. Supported near precision/recall is a diagnostic proxy rather than visibility-aware operating truth. Unsupported relative references fall back to the common0.7 benchmark; no environmental normality is fabricated.

The e-means study uses5CPU NumPy passes and reassigns every vector to its nearest final centroid. The official documentation distinguishes last-epoch labels from final-centroid assignments. This small study does not claim the paper's large-scale10x speedup.

The frozen policy scripts choose their candidate from **BO history only**; matched random search is a control, unlike quality HPO which selects over the evaluated union. For f1 RGB-weather review, random search reached Dev loss0.37581 versus BO0.39159 and common0.39474, but was not eligible under that selection rule. Thus neither BO superiority nor the absence of any useful relative policy is established. Keep the frozen artifacts unchanged. A future separately registered study should select over the BO/random union and isolate balance reservation from threshold changes before opening a new confirmation cohort.
