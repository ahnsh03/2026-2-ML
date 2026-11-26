"""Evaluate the frozen version-paired GP, environment and review policy on heldout.

All target values in this script are offline diagnostics. The portable review
component consumes only image/output features and checkpoint-bound models.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import joblib
import numpy as np
from scipy.special import ndtr
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from quality_estimation.versioned_review import VersionedReview
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.diagnostic_targets import auxiliary_targets
from quality_estimation.cluster_review import select_review
from quality_estimation.spatial_identity import validate_pack_perception
from prepare import dump
sys.path.insert(0, str(Path(__file__).parents[1] / 'roi9_gp_20261125'))
from study import load


def macro(values, roads):
    return np.nanmean([np.nanmean(values[roads == road], axis=0) for road in np.unique(roads)], axis=0)


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def point_metrics(predicted, truth, roads):
    return dict(rows=len(truth), macro_road_mae=np.asarray(macro(abs(predicted - truth), roads)).tolist(),
                micro_mae=np.nanmean(abs(predicted - truth), axis=0).tolist())


def review_metrics(ix, picked, truth, roads):
    local = np.isin(ix, picked)
    positive = truth[ix] < .7
    road_recall = [float(local[(roads[ix] == road) & positive].mean()) for road in np.unique(roads[ix])
                   if ((roads[ix] == road) & positive).sum()]
    found = int(positive[local].sum())
    return dict(population=len(ix), positives=int(positive.sum()), budget=int(local.sum()), low_iou_found=found,
                precision=float(found / local.sum()) if local.any() else None,
                recall=float(found / positive.sum()) if positive.any() else None,
                macro_road_recall=float(np.mean(road_recall)) if road_recall else None,
                benchmark_nonlow_selected=int((~positive[local]).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    study = args.study
    if not (study / 'protocol/heldout_evaluation_frozen.json').exists():
        raise ValueError('development model/policy choices must be frozen first')
    review = VersionedReview(study / 'g1_quality/bundle')
    pack, schema = load(study / 'g1_quality/packs', ['heldout_diagnostic'])
    receipt = json.loads((study / 'g1_quality/packs/heldout_diagnostic.json').read_text())
    validate_pack_perception(receipt, review.spec['perception_identity'], review.spec['perception_weights'])
    f0, f0_schema = load(study / 'f0_quality/packs', ['heldout_diagnostic'])
    assert np.array_equal(pack['ids'], f0['ids']) and np.array_equal(pack['pixel_sha256'], f0['pixel_sha256'])
    if receipt['index_sha256'] != hashlib.sha256((study / 'protocol/quality_index_frozen.json').read_bytes()).hexdigest():
        raise ValueError('heldout membership changed')
    x = feature_matrix(pack, schema, review.columns)
    env_x = feature_matrix(pack, schema, review.environment.columns)
    prediction = review.predict(x, env_x)
    truth = auxiliary_targets(pack['counts'], pack['y'])[:, :len(review.target_names)]
    base_gp = SpatialQualityGP.load(study.parent / 'ml_roi9_gp_20261125/frozen_bundle_final/quality_spatial_gp.npz')
    base_columns = base_gp.feature_names_.tolist()
    base_mean, _ = base_gp.predict(feature_matrix(f0, f0_schema, base_columns))
    # Deliberately mismatched versions only as an explicitly named offline
    # diagnostic. This is never exported as an accepted model/version pair.
    stale_mean, _ = base_gp.predict(feature_matrix(pack, schema, base_columns))
    frozen = json.loads((study / 'g1_quality/refit/quality_selection_frozen.json').read_text())
    rf_record = frozen['rf_comparator']
    rf = joblib.load(study / 'g1_quality/refit' / rf_record['model'])
    rf_mean = rf.predict(feature_matrix(pack, schema, rf_record['columns']))
    sites = pack['site']
    primary = np.array([str(site).startswith('confirm26_') for site in sites])
    shift = np.array([str(site).startswith('drive26_') and str(cohort).find('shift') >= 0
                      for site, cohort in zip(sites, pack['cohort'])])
    # Source membership, rather than a free-form cohort string, defines cohorts.
    index = json.loads((study / 'protocol/quality_index_frozen.json').read_text())['heldout_diagnostic']['rows']
    sources = np.array([r['source_key'] for r in index])
    primary = np.char.startswith(sources, 'drive_confirmation_test_')
    shift = np.char.startswith(sources, 'drive_calibration_shift_')
    main_mask = np.char.startswith(sources, 'drive_test_')
    assert primary.sum() == 366 and main_mask.sum() == 366 and shift.sum() == 102
    counts = pack['counts']
    gt_pixels = (counts[:, :, 0] + counts[:, :, 2]).sum(1)
    ng = (counts[:, 6:8, 0] + counts[:, 6:8, 2]).sum(1)
    fg = (counts[:, 4:6, 0] + counts[:, 4:6, 2]).sum(1)
    point = {}
    for name, mask in [('confirmation', primary), ('main', main_mask), ('condition_shift', shift)]:
        valid = mask & np.isfinite(truth[:, 0])
        point[name] = dict(g1_gp=point_metrics(np.clip(prediction['raw_mean'][valid], 0, 1), truth[valid], sites[valid]),
                          g1_rf=point_metrics(rf_mean[valid, None], truth[valid, :1], sites[valid]),
                          g0_on_f0=point_metrics(np.clip(base_mean[valid, :1], 0, 1), f0['y'][valid, None], sites[valid]),
                          stale_g0_on_f1=point_metrics(np.clip(stale_mean[valid, :1], 0, 1), truth[valid, :1], sites[valid]))
    risk = {}
    mean, cov = prediction['risk_mean'], prediction['risk_covariance']
    for name, mask in [('confirmation', primary), ('main', main_mask), ('condition_shift', shift)]:
        supported = mask & (gt_pixels >= 100) & np.isfinite(truth[:, 0])
        ix = np.flatnonzero(supported)
        low = truth[ix, 0] < .7
        probability = prediction['low_quality_probability'][ix]
        sigma = prediction['quality_epistemic_sd'][ix]
        sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 1e-12))
        target_valid = np.isfinite(truth) & (ng >= 50)[:, None] & (fg >= 50)[:, None]
        target_valid[:, 0] = np.isfinite(truth[:, 0]) & (gt_pixels >= 100)
        coverages = []
        for j in range(len(review.target_names)):
            valid = mask & target_valid[:, j]
            hit = abs(truth[valid, j] - mean[valid, j]) <= 1.6448536269514722 * sd[valid, j]
            coverages.append(dict(target=review.target_names[j], rows=int(valid.sum()),
                                  coverage_90=float(hit.mean()) if len(hit) else None,
                                  mean_width_90=float((3.2897072539029444 * sd[valid, j]).mean()) if len(hit) else None))
        bins = []
        for left in np.arange(0, 1, .2):
            chosen = (probability >= left) & (probability < left + .2 if left < .79 else probability <= 1)
            bins.append(dict(left=float(left), rows=int(chosen.sum()),
                             mean_probability=float(probability[chosen].mean()) if chosen.any() else None,
                             actual_low_fraction=float(low[chosen].mean()) if chosen.any() else None))
        aux_valid = mask & (ng >= 50) & (fg >= 50) & np.isfinite(truth[:, 1:]).all(1)
        near_index = 3 if truth.shape[1] == 4 else 1
        states = (truth[aux_valid, near_index] < .8).astype(int) + 2 * (truth[aux_valid, 2] < .5).astype(int)
        # probability order: both retained, near weak only, far weak only, both weak
        band_probability = prediction['band_diagnostic_probabilities'][aux_valid]
        joint = dict(rows=int(aux_valid.sum()), target='near F1' if near_index == 3 else 'near recall only; precision not certified',
                     state_counts=np.bincount(states, minlength=4).tolist(),
                     multiclass_brier=float(np.mean(np.sum((band_probability - np.eye(4)[states]) ** 2, axis=1))) if len(states) else None,
                     accuracy=float(np.mean(band_probability.argmax(1) == states)) if len(states) else None)
        risk[name] = dict(rows=len(ix), positives=int(low.sum()),
             macro_road_brier=float(macro((probability - low) ** 2, sites[ix])) if len(ix) else None,
             quality_probability_auc=float(roc_auc_score(low, probability)) if len(np.unique(low)) == 2 else None,
             sigma_only_auc=float(roc_auc_score(low, sigma)) if len(np.unique(low)) == 2 else None,
             sigma_quality_spearman=float(spearmanr(sigma, truth[ix, 0]).statistic) if len(ix) and np.ptp(sigma) else None,
             interval_coverage=coverages, reliability=bins, joint_band_diagnostics=joint)
    review_table = {}
    selections = {}
    policies = ['g1_policy', 'global_probability', 'global_probability_audit', 'gp_low_mean', 'rf_low_mean', 'sigma_only']
    for cohort_name, mask in [('confirmation', primary), ('main', main_mask), ('condition_shift', shift)]:
        ix = np.flatnonzero(mask & (gt_pixels >= 100) & np.isfinite(truth[:, 0]))
        rows = []
        for fraction in [.05, .1, .2, 3072 / 7442]:
            for policy in policies:
                selected = []
                for road in sorted(set(sites[ix])):
                    local_ix = ix[sites[ix] == road]
                    budget = int(np.ceil(len(local_ix) * fraction))
                    if policy == 'g1_policy':
                        sub = {key: value[local_ix] for key, value in prediction.items()}
                        picked, _ = review.select(pack['ids'][local_ix], sub, budget)
                    else:
                        score = {'global_probability': prediction['low_quality_probability'],
                                 'global_probability_audit': prediction['low_quality_probability'],
                                 'gp_low_mean': 1 - prediction['estimated_iou'],
                                 'rf_low_mean': 1 - rf_mean,
                                 'sigma_only': prediction['quality_epistemic_sd']}[policy]
                        picked = select_review(pack['ids'][local_ix], score[local_ix], np.zeros(len(local_ix), int), budget,
                                               audit_fraction=.1 if policy == 'global_probability_audit' else 0.)
                    selected.extend(local_ix[picked].tolist())
                rows.append(dict(policy=policy, fraction=fraction, **review_metrics(ix, selected, truth[:, 0], sites)))
                if cohort_name == 'confirmation' and fraction == .1:
                    selections[policy] = pack['ids'][selected].tolist()
            random_metrics = []
            for seed in range(100):
                rng = np.random.default_rng(seed)
                selected = []
                for road in sorted(set(sites[ix])):
                    local_ix = ix[sites[ix] == road]
                    selected.extend(rng.choice(local_ix, int(np.ceil(len(local_ix) * fraction)), replace=False).tolist())
                random_metrics.append(review_metrics(ix, selected, truth[:, 0], sites))
            rows.append(dict(policy='random_100_seeds', fraction=fraction,
                 mean={key: float(np.mean([m[key] for m in random_metrics])) for key in ['budget', 'low_iou_found', 'precision', 'recall', 'macro_road_recall'] if random_metrics[0][key] is not None},
                 low_iou_found_range=[min(m['low_iou_found'] for m in random_metrics), max(m['low_iou_found'] for m in random_metrics)]))
        review_table[cohort_name] = rows
    environment = []
    for cluster in range(len(review.environment.centers)):
        mask = primary & (prediction['environment_cluster'] == cluster)
        environment.append(dict(cluster=cluster, rows=int(mask.sum()),
             roads=len(set(sites[mask])), weather=sorted(set(pack['weather'][mask].tolist())),
             hour=sorted(set(pack['hour'][mask].tolist())),
             ood_rows=int(prediction['environment_out_of_reference'][mask].sum()),
             mean_actual_iou=float(np.nanmean(truth[mask, 0])) if mask.any() else None))
    gates = json.loads((study / 'protocol/evaluation_gate_v2_frozen.json').read_text())['quality_adoption']
    primary_gp = point['confirmation']['g1_gp']['macro_road_mae'][0]
    primary_g0 = point['confirmation']['g0_on_f0']['macro_road_mae'][0]
    quality_pass = bool(frozen['gp_development_acceptable_to_rf'] and primary_gp <= primary_g0 + gates['primary_heldout_mae_increase_tolerance_vs_g0_f0'])
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        model_perception_identity=review.spec['perception_identity'], point=point, risk=risk,
        same_road_budget_review=review_table, environment_confirmation=environment,
        quality_adoption_gate_passes=quality_pass, policy=review.policy,
        limitations=['IoU<0.7 is a common evaluation benchmark, not operating safety/visibility truth.',
                     'Gaussian intervals are empirical road-calibrated intervals, without OOD coverage guarantee.',
                     'Near/far auxiliary targets use image bands, not metric-distance zones.',
                     'All selected candidates still require verified GT and explicit training admission.',
                     'Stale g0 on f1 is an intentional offline mismatch diagnostic, not a usable deployed pair.',
                     'No feature, policy, or threshold is selected or retuned on heldout.'])
    dump(study / 'reports/quality_heldout_assessment.json', finite_json(output))
    dump(study / 'reports/heldout_review_sample_ids_private.json', selections)
    np.savez_compressed(study / 'reports/g1_heldout_predictions.npz', ids=pack['ids'], Y=truth,
                        rf_mean=rf_mean, **prediction)
    print('HELDOUT QUALITY', primary_gp, 'g0/f0', primary_g0, 'gate', quality_pass, flush=True)


if __name__ == '__main__':
    main()
