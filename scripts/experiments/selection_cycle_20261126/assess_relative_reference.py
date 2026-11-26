"""Evaluate the version-frozen relative-reference proposal without changing runtime."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import numpy as np
from scipy.special import ndtr
from quality_estimation.environment_references import reference_thresholds
from quality_estimation.diagnostic_targets import gt_review_flags
from quality_estimation.cluster_review import mixture_priority
from quality_estimation.versioned_review import VersionedReview
from prepare import dump
from versioned_relative_study import objective
sys.path.insert(0, str(Path(__file__).parents[1] / 'roi9_gp_20261125'))
from study import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    study = parser.parse_args().study
    reference_path = study / 'g1_quality/relative_reference/relative_study_frozen.json'
    frozen = json.loads(reference_path.read_text())
    review = VersionedReview(study / 'g1_quality/bundle')
    if frozen['perception_identity'] != review.spec['perception_identity']:
        raise ValueError('environment reference belongs to another perception version')
    for name, field in [('quality_gp.npz', 'model_sha256'), ('environment.json', 'environment_sha256')]:
        if hashlib.sha256((study / 'g1_quality/bundle' / name).read_bytes()).hexdigest() != frozen[field]:
            raise ValueError('relative-reference artifact version differs')
    pack, _ = load(study / 'g1_quality/packs', ['heldout_diagnostic'])
    prediction = np.load(study / 'reports/g1_heldout_predictions.npz')
    assert np.array_equal(pack['ids'], prediction['ids'])
    rows = json.loads((study / 'protocol/quality_index_frozen.json').read_text())['heldout_diagnostic']['rows']
    source = np.array([r['source_key'] for r in rows])
    flags = [gt_review_flags(c, float(q) if np.isfinite(q) else None) for c,q in zip(pack['counts'], pack['y'])]
    harm = np.array([f['near_precision_or_recall_reduced'] is True for f in flags])
    retained = np.array([f['near_precision_and_recall_retained'] is True for f in flags])
    assessment = {}
    for name, prefix in [('confirmation', 'drive_confirmation_test_'), ('main', 'drive_test_'), ('condition_shift', 'drive_calibration_shift_')]:
        ix = np.flatnonzero(np.char.startswith(source, prefix))
        labels = prediction['environment_cluster'][ix]
        mu = prediction['risk_mean'][ix, 0]
        sd = np.sqrt(np.maximum(prediction['risk_covariance'][ix, 0, 0], 1e-12))
        sigma = prediction['quality_epistemic_sd'][ix]
        ood = prediction['environment_out_of_reference'][ix]
        selected = frozen['selected']
        tau = reference_thresholds(selected['reference'], labels, ood=ood)
        parameters = selected['parameters']
        relative_score = mixture_priority(ndtr((tau-mu)/sd), sigma, parameters['uncertainty_weight'])
        relative_loss, relative_detail = objective(pack['ids'][ix], pack['site'][ix], labels, harm[ix], retained[ix], relative_score, parameters['quota_exponent'])
        baseline = frozen['common_best']
        common_score = mixture_priority(ndtr((.7-mu)/sd), sigma, baseline['uncertainty_weight'])
        common_loss, common_detail = objective(pack['ids'][ix], pack['site'][ix], labels, harm[ix], retained[ix], common_score, baseline['quota_exponent'])
        assessment[name] = dict(rows=len(ix), near_defects=int(harm[ix].sum()),
             relative_loss=relative_loss, common_loss=common_loss, improvement=common_loss-relative_loss,
             relative_budgets=relative_detail, common_budgets=common_detail,
             noncommon_threshold_rows=int((tau != .7).sum()), ood_rows=int(ood.sum()))
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(), perception_identity=frozen['perception_identity'],
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        development_gain_gate=frozen['diagnostic_gain_gate_passes'], heldout=assessment,
        relative_runtime_adopted=False, absolute_failure_benchmark=.7,
        scope='Fixed proposal diagnostic only. No visibility-normal labels or extra retraining arm validate an environmental pass criterion. The primary runtime remains the separately evaluated common-IoU policy.')
    dump(study / 'reports/relative_reference_heldout_assessment.json', output)
    print('RELATIVE HELDOUT DIAGNOSTIC', assessment['confirmation']['improvement'], flush=True)


if __name__ == '__main__':
    main()
