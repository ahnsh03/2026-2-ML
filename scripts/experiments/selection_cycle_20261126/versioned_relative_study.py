"""Refit environment-relative review references to each exact perception version.

This is a separate near-defect retrieval diagnostic. It never relabels an
absolute-low-IoU frame as a pass or changes the primary versioned runtime policy.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import itertools
import json
import sys
import numpy as np
from scipy.special import ndtr
from quality_estimation.versioned_review import VersionedReview
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_identity import validate_pack_perception
from quality_estimation.environment_references import fit_relative_references, reference_thresholds
from quality_estimation.cluster_review import select_review, mixture_priority
from quality_estimation.diagnostic_targets import gt_review_flags
from quality_estimation.bounded_bo import next_expected_improvement
from prepare import dump
sys.path.insert(0, str(Path(__file__).parents[1] / 'roi9_gp_20261125'))
from study import load


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def objective(ids, sites, clusters, harm, retained, priority, quota, fractions=(.1, 3072/7442)):
    details = []
    for fraction in fractions:
        picked = np.zeros(len(ids), bool)
        for road in np.unique(sites):
            ix = np.flatnonzero(sites == road)
            selected = select_review(ids[ix], priority[ix], clusters[ix], int(np.ceil(len(ix) * fraction)),
                                     quota_exponent=quota, audit_fraction=.1, audit_rounding='ceil')
            picked[ix[selected]] = True
        def recall(groups):
            values = [picked[(groups == g) & harm].mean() for g in np.unique(groups) if ((groups == g) & harm).sum() >= 3]
            return float(np.mean(values)) if values else 0.
        road_recall, cluster_recall = recall(sites), recall(clusters)
        retained_cost = float(retained[picked].mean())
        loss = .6 * (1 - road_recall) + .2 * (1 - cluster_recall) + .2 * retained_cost
        details.append(dict(fraction=fraction, budget=int(picked.sum()), near_defects_found=int((picked & harm).sum()),
                            macro_road_near_defect_recall=road_recall, macro_cluster_near_defect_recall=cluster_recall,
                            near_retained_review_fraction=retained_cost, loss=loss))
    return float(np.mean([v['loss'] for v in details])), details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['packs', 'bundle', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise ValueError('versioned relative study output exists')
    output.mkdir(parents=True)
    review = VersionedReview(args.bundle)
    data = {}
    schema = None
    for role in ['calibration', 'development']:
        data[role], schema = load(args.packs, [role])
        validate_pack_perception(json.loads((args.packs / (role + '.json')).read_text()),
                                 review.spec['perception_identity'], review.spec['perception_weights'])
    cal, dev = data['calibration'], data['development']
    def counts_flags(pack):
        flags = [gt_review_flags(c, float(q) if np.isfinite(q) else None) for c, q in zip(pack['counts'], pack['y'])]
        return flags
    cf, df = counts_flags(cal), counts_flags(dev)
    eligible = np.array([v['valid_for_low_iou_benchmark'] for v in cf])
    near_strong = np.array([v['near_precision_and_recall_retained'] is True for v in cf])
    # This is an observed precision/recall reference, not a visibility-normal GT.
    ec = review.environment.assign(feature_matrix(cal, schema, review.environment.columns))
    ed = review.environment.assign(feature_matrix(dev, schema, review.environment.columns))
    primary = np.array([str(v).startswith('drive26_') for v in dev['site']])
    ix = np.flatnonzero(primary)
    prediction = review.predict(feature_matrix(dev, schema, review.columns), feature_matrix(dev, schema, review.environment.columns))
    harm = np.array([v['near_precision_or_recall_reduced'] is True for v in df])[ix]
    retained = np.array([v['near_precision_and_recall_retained'] is True for v in df])[ix]
    ids, sites, labels = dev['ids'][ix], dev['site'][ix], ed['environment_cluster'][ix]
    mu = prediction['risk_mean'][ix, 0]
    sd = np.sqrt(np.maximum(prediction['risk_covariance'][ix, 0, 0], 1e-12))
    sigma = prediction['quality_epistemic_sd'][ix]
    ood = ed['environment_out_of_reference'][ix]
    parameters = [dict(reference_population=population, quantile=q, shrinkage=s, quota_exponent=e, uncertainty_weight=u)
                  for population, q, s, e, u in itertools.product(['all_supported', 'near_retained'], [.1, .25, .4], [25., 100., 400.], [0., .5, 1.], [0., .25])]
    coordinates = np.clip(np.array([[int(v['reference_population'] == 'near_retained'), (v['quantile'] - .1) / .3,
                             np.log(v['shrinkage']/25) / np.log(16), v['quota_exponent'], v['uncertainty_weight']/.25] for v in parameters]),0.,1.)
    common = []
    for quota, uncertainty in itertools.product([0., .5, 1.], [0., .25]):
        priority = mixture_priority(ndtr((.7-mu)/sd), sigma, uncertainty)
        loss, details = objective(ids, sites, labels, harm, retained, priority, quota)
        common.append(dict(quota_exponent=quota, uncertainty_weight=uncertainty, loss=loss, budgets=details))
    trials = {}
    def evaluate(i):
        if i in trials:
            return trials[i]['loss']
        v = parameters[i]
        mask = eligible & (near_strong if v['reference_population'] == 'near_retained' else True)
        reference = fit_relative_references(cal['y'][mask], ec['environment_cluster'][mask], cal['site'][mask], cal['xy'][mask],
                                             quantile=v['quantile'], shrinkage=v['shrinkage'])
        threshold = reference_thresholds(reference, labels, ood=ood)
        priority = mixture_priority(ndtr((threshold-mu)/sd), sigma, v['uncertainty_weight'])
        loss, details = objective(ids, sites, labels, harm, retained, priority, v['quota_exponent'])
        trials[i] = dict(parameters=v, loss=loss, budgets=details, reference=reference,
                          noncommon_development_rows=int(np.count_nonzero(threshold != .7)))
        return loss
    initial = [0, 53, 54, 107]
    history, losses = [], []
    for _ in range(16):
        i = next_expected_improvement(coordinates, history, losses, initial)
        history.append(i)
        losses.append(evaluate(i))
    random = initial + np.random.default_rng(261126).permutation([i for i in range(len(parameters)) if i not in initial]).tolist()[:12]
    random_losses = [evaluate(int(i)) for i in random]
    best = min(set(history+random), key=lambda i: (trials[i]['loss'], -parameters[i]['shrinkage']))
    common_best = min(common, key=lambda v: v['loss'])
    improvement = common_best['loss'] - trials[best]['loss']
    support = bool(harm.sum() >= 20 and len(set(sites[harm])) >= 2)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        perception_identity=review.spec['perception_identity'],
        model_sha256=sha(args.bundle/'quality_gp.npz'), environment_sha256=sha(args.bundle/'environment.json'),
        calibration_rows=len(cal['ids']), calibration_supported_rows=int(eligible.sum()),
        calibration_near_retained_rows=int((eligible&near_strong).sum()),
        development_rows=len(ix), development_near_defects=int(harm.sum()),
        selected=trials[best], common_best=common_best, development_loss_improvement=improvement,
        diagnostic_gain_gate_passes=bool(support and improvement >= .01),
        primary_runtime_policy_changed=False, absolute_quality_threshold=.7,
        bo_history=history, bo_losses=losses, random_history=random, random_losses=random_losses,
        shared_initial=initial, budget_each=16, trials={str(k):v for k,v in trials.items()},
        common_comparators=common, source_sha256=sha(__file__),
        interpretation='Version-specific near-defect retrieval diagnostic; normal visibility is unlabelled. No low absolute IoU becomes a pass. Primary runtime stays the separately frozen absolute-quality policy; any relative proposal requires heldout and next-cycle training validation.')
    dump(output/'relative_study_frozen.json', result)
    (output/'_SUCCESS').write_text(sha(output/'relative_study_frozen.json')+'\n')
    print('VERSIONED RELATIVE', improvement, 'diagnostic gate', result['diagnostic_gain_gate_passes'], flush=True)


if __name__ == '__main__':
    main()
