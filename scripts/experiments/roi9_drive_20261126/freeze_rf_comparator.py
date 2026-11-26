"""Freeze same-membership RF controls before prospective test collection."""
import argparse,json,joblib
from pathlib import Path
from datetime import datetime,timezone
from sklearn.ensemble import RandomForestRegressor
from quality_estimation.portable_forest import PortableForest
from quality_estimation.region_feature_views import feature_matrix
from feature_study import old_training,read,save,sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['old','study','feature-study','output']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 assert not any(read(p)['role']=='test' for p in (a.study/'runs').glob('*/summary.json'))
 selected=read(a.feature_study/'selection_frozen.json');cols=selected['chosen']['columns'];record=next(r for r in selected['results'] if r['kind']=='RF' and r['columns']==cols)
 train,schema=old_training(a.old);primary=joblib.load(a.feature_study/record['model']);aux=joblib.load(a.feature_study/record['auxiliary']);joblib.dump(aux,a.output/'primary_auxiliary.joblib',compress=2)
 context={k:selected['chosen'][k] for k in []};context.update(profile='roi9-front-v1',regions_sha256=read(a.old/'frozen_bundle_final/feature_recipe.json')['regions_sha256'],checkpoint_sha256=read(a.old/'frozen_bundle_final/feature_recipe.json')['checkpoint_sha256'])
 portable=PortableForest.from_sklearn(primary,cols,context);portable.save(a.output/'primary.npz')
 far_record=selected.get('far_auxiliary');far=None
 if far_record:
  far=RandomForestRegressor(**record['chosen']['parameters'],random_state=23411,n_jobs=2).fit(feature_matrix(train,schema,far_record['columns']),train['Y'][:,2])
  PortableForest.from_sklearn(far,far_record['columns'],context).save(a.output/'far.npz')
 save(a.output/'frozen.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),selection_sha256=sha(a.feature_study/'selection_frozen.json'),membership_sha256=sha(a.old/'frozen_bundle_final/training_membership.json'),training_rows=len(train['ids']),primary_columns=cols,far_columns=far_record['columns'] if far_record else None,parameters=record['chosen']['parameters'],artifacts={p.name:sha(p) for p in a.output.iterdir()},scope='RF single-output IoU, multi-output near control, and separate far RF on same31 columns/membership as farGP. No test selection or RF uncertainty claim.'))
 print('RF controls frozen',len(train['ids']),flush=True)
if __name__=='__main__':main()
