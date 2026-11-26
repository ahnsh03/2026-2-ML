"""Keep acquisition roles immutable while recording current label/model exposure.

The supported followup was retired into calibration. Entire related road
cohorts, not merely thinned fitted IDs, must not be reused as unseen tests.
Perception training admission is a separate, unapproved decision.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,copy
from feature_study import read,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['study','old','registry','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();n=a.study;a.output.mkdir(parents=True,exist_ok=False)
    index=read(a.registry/'index_full.json');annotated=copy.deepcopy(index)
    train={r['sample_id'] for r in read(a.old/'frozen_bundle_final/training_membership.json')['rows']}
    calibration=set(read(n/'calibration_v2_final/membership.json')['ids']);lookup={};counts=Counter();members=Counter()
    for key,pack in annotated.items():
        for row in pack['rows']:
            sid=row['sample_id'];cohort=row.get('cohort');row['acquisition_cohort']=cohort
            if sid in train:role='f0_quality_training'
            elif cohort=='development':role='quality_development'
            elif cohort=='calibration':role='calibration_source_related'
            elif cohort=='support_test':role='retired_supported_followup_calibration_related'
            elif cohort=='test':role='opened_primary_test'
            elif cohort=='confirmation_test':role='opened_confirmation_test'
            elif cohort in ['condition_shift','calibration_shift']:role='opened_condition_shift'
            else:role='historical_exposure_review_required'
            row.update(current_quality_role=role,spatial_calibration_v2_member=sid in calibration,reusable_as_unseen_test=False,perception_training_admission_approved=False)
            if sid in lookup:raise ValueError('duplicate current sample role')
            lookup[sid]=dict(pack=key,current_quality_role=role,spatial_calibration_v2_member=sid in calibration)
            counts[role]+=1;members[role]+=int(sid in calibration)
    assert train<=lookup.keys() and calibration<=lookup.keys()
    assert len(calibration)==612
    assert all(lookup[sid]['current_quality_role'] in ['calibration_source_related','retired_supported_followup_calibration_related'] for sid in calibration)
    assert not train&calibration
    save(a.output/'index_with_current_roles.json',annotated)
    save(a.output/'roles_by_sample.json',lookup)
    save(a.output/'summary.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),parent_index_sha256=sha(a.registry/'index_full.json'),current_index_sha256=sha(a.output/'index_with_current_roles.json'),quality_training_membership_sha256=sha(a.old/'frozen_bundle_final/training_membership.json'),spatial_calibration_membership_sha256=sha(n/'calibration_v2_final/membership.json'),rows=len(lookup),current_role_counts=dict(counts),calibration_members_by_role=dict(members),perception_training_admission_approved=0,scope=__doc__))


if __name__=='__main__':main()
