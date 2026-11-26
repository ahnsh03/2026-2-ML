"""Apply previously recorded diagnostic gates, never refit on confirmation."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
from feature_study import read,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();n=a.study
    protocol=read(n/'confirmation_acceptance_before_scores.json')
    assert sha(n/'confirmation/freeze_receipt.json')==protocol['confirmation_seal_sha256']
    r=read(n/'confirmation/evaluation/results.json')
    assert protocol['created_utc']<r['created_utc']
    packs=list((n/'confirmation/packs').glob('*.npz'))
    assert packs and all(datetime.fromisoformat(protocol['created_utc']).timestamp()<p.stat().st_mtime for p in packs)
    reg=r['regression'];old=reg['GP28_primary'];far=reg['GP28_plus_far31']
    gain=old['macro_mae'][2]-far['macro_mae'][2]
    better=sum(far['byroad'][road][2]<=v[2] for road,v in old['byroad'].items())
    metrics=r['calibration'];v2=metrics['selected_calibration'];raw=metrics['raw_composed'];v1=metrics['previous_v1']
    gate=protocol['calibration_v2'];checks={}
    for target,key in [('near_recall','minimum_near_coverage90'),('far_recall','minimum_far_coverage90')]:
        coverage=v2['marginals'][target]['coverage90']
        checks[target+'_minimum_coverage']=coverage>=gate[key]
        checks[target+'_coverage_vs_raw']=coverage>=raw['marginals'][target]['coverage90']-gate['maximum_coverage_drop_vs_raw']
    checks['NLL_vs_v1']=v2['spatial']['nll']<=v1['spatial']['nll']+gate['maximum_spatial_NLL_increase_vs_v1']
    checks['Brier_vs_v1']=v2['spatial']['brier']<=v1['spatial']['brier']+gate['maximum_spatial_Brier_increase_vs_v1']
    save(a.output,dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(n/'confirmation_acceptance_before_scores.json'),evaluation_sha256=sha(n/'confirmation/evaluation/results.json'),far_auxiliary=dict(macro_mae_gain=gain,roads_nonworse=better,roads=len(old['byroad']),diagnostic_gate_passed=gain>=protocol['far_auxiliary']['minimum_road_macro_MAE_improvement'] and better>=protocol['far_auxiliary']['minimum_nonworse_roads']),spatial_calibration=dict(checks=checks,diagnostic_gate_passed=all(checks.values()),metrics=v2,previous_v1=v1,raw=raw),automatic_admission=False,operational_failure_classifier_validated=False,scope='Same-map conditional road confirmation. Diagnostic gates do not establish every spatial class, environment-specific failure truth, independent-map validity or perception retraining gain.'))


if __name__=='__main__':main()
