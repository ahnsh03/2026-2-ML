"""Freeze GT-free risk and environment-matched random admissions before GT joining."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse
import json
import numpy as np
from common.bridge import sha256_file
from quality_estimation.support_aware_review import SupportAwareReview
from quality_estimation.perception_identity import identity_from_adaptation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','feature-only-pool','rgb-only-manifest','parent-membership','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--budget',type=int,default=2048);p.add_argument('--seed',type=int,default=281128);a=p.parse_args()
    if a.output.exists():raise ValueError('admission output exists')
    review=SupportAwareReview(a.bundle)
    with np.load(a.feature_only_pool)as z:
        allowed={'X','environment_X','ids','ordered_features','environment_ordered_features'}
        if set(z.files)!=allowed:raise ValueError('selection pool must contain features only; GT fields are forbidden')
        x=z['X'];ex=z['environment_X'];ids=z['ids']
        if z['ordered_features'].tolist()!=review.columns or z['environment_ordered_features'].tolist()!=review.environment.columns:
            raise ValueError('feature-only pool order differs from review models')
    rows=[json.loads(line)for line in a.rgb_only_manifest.read_text().splitlines()if line.strip()]
    forbidden={'semantic','mask','ground_truth','y','counts','gt_pixels','scene_gt_class'}
    if any(forbidden&set(row)for row in rows):raise ValueError('RGB-only selector manifest contains GT fields')
    if [r['sample_id']for r in rows]!=ids.tolist():raise ValueError('RGB-only selector membership differs')
    if a.parent_membership.name!='training_membership.json'or identity_from_adaptation(a.parent_membership.parent)!=review.identity:
        raise ValueError('selection parent lineage differs from review bundle')
    members=json.loads(a.parent_membership.read_text());seen={r['sample_id']for r in members};pixels={r['pixel_sha256']for r in members}
    if seen&set(ids)or pixels&{r['pixel_sha256']for r in rows}:raise ValueError('already admitted candidate reused')
    prediction=review.predict(x,ex);selected,priority,details=review.select_with_features(ids,prediction,a.budget,x,seed=a.seed)
    # Match environment/source availability exactly. No candidate outcome is used.
    def stratum(row):
        h=int(row['hour']);light=0 if 8<=h<18 else 1 if 6<=h<8 or 18<=h<20 else 2
        return str((int(row['weather']),light,row['pool_origin']))
    strata=defaultdict(list)
    for i,row in enumerate(rows):strata[stratum(row)].append(i)
    quotas=Counter(stratum(rows[i])for i in selected);rng=np.random.default_rng(a.seed);random=[]
    for group,count in sorted(quotas.items()):random+=rng.choice(strata[group],count,replace=False).tolist()
    a.output.mkdir(parents=True)
    for name,chosen in [('Risk2048',selected),('Random2048',random)]:
        admissions=[]
        for j,i in enumerate(chosen):
            record=dict(rows[int(i)],selection_rank=j+1,perception_identity=review.identity,admission_approved=False,uses_ground_truth=False)
            if name=='Risk2048':record.update(selection_component=details['reasons'][j],conditional_inclusion_probability=details['conditional_inclusion_probability'][j],priority=float(priority[i]))
            else:record.update(selection_component='environment_matched_random',conditional_inclusion_probability=quotas[stratum(rows[i])]/len(strata[stratum(rows[i])]))
            admissions.append(record)
        (a.output/(name+'_rgb_admissions.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n'for r in admissions))
    np.savez_compressed(a.output/'feature_only_predictions.npz',ids=ids,**prediction)
    report=dict(version='support-aware-matched-admissions-1',candidate_rows=len(rows),bank_rows=a.budget,
        budget_fraction=a.budget/len(rows),source_bundle_sha256=sha256_file(a.bundle/'bundle_manifest.json'),
        feature_only_pool_sha256=sha256_file(a.feature_only_pool),rgb_only_manifest_sha256=sha256_file(a.rgb_only_manifest),
        parent_membership_sha256=sha256_file(a.parent_membership),perception_identity=review.identity,
        matched_environment_source_counts=dict(quotas),risk_random_overlap=len(set(selected)&set(random)),
        parent_training_overlap=0,uses_candidate_gt_for_selection=False,source_sha256=sha256_file(Path(__file__)),
        scope='Closed offline buffer. Random arm matches risk-arm SDK weather/lighting/source quotas. Both start from the same parent with equal replay and updates; overlap is allowed and reported. GT join is a later operation.')
    (a.output/'selection.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (a.output/'_SUCCESS').write_text(sha256_file(a.output/'selection.json')+'\n')
    print('FROZEN MATCHED BANKS',report['candidate_rows'],report['bank_rows'],report['risk_random_overlap'],flush=True)


if __name__=='__main__':main()
