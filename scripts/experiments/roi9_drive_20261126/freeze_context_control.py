"""Freeze context-only diagnostic controls before prospective test collection.

The fixed-kernel RGB control is an input ablation, not a separately tuned winner.
Weather/hour medians use simulator metadata only as an offline oracle control.
"""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from feature_study import old_training,bind,read,save,sha
from calibrate import environment_profile,assign_environment

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['old','study','output']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 assert not any(read(p)['role']=='test' for p in (a.study/'runs').glob('*/summary.json'))
 data,schema=old_training(a.old);profile=environment_profile(a.old,data,schema);columns=profile['ordered_features']
 recipe=read(a.old/'frozen_bundle_final/feature_recipe.json');params=recipe['tasks']['quality_spatial_gp']['parameters']
 model=bind(SpatialQualityGP(**params).fit(feature_matrix(data,schema,columns),data['Y']),columns,recipe);model.save(a.output/'rgb54_GP.npz')
 fine,distance,coarse=assign_environment(data,schema,profile)
 contexts={}
 for w,h in sorted(set(zip(data['weather'],data['hour']))):
  ix=(data['weather']==w)&(data['hour']==h);contexts[f'{w}:{h}']=dict(median=float(np.median(data['Y'][ix,0])),rows=int(ix.sum()))
 clusters={str(int(k)):dict(median=float(np.median(data['Y'][fine==k,0])),rows=int((fine==k).sum())) for k in np.unique(fine)}
 save(a.output/'environment_profile.json',profile)
 save(a.output/'frozen.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),model_sha256=sha(a.output/'rgb54_GP.npz'),membership_sha256=sha(a.old/'frozen_bundle_final/training_membership.json'),ordered_features=columns,parameters=params,metadata_medians=contexts,cluster_medians=clusters,fallback_median=float(np.median(data['Y'][:,0])),profile_sha256=sha(a.output/'environment_profile.json'),test_not_collected=True,scope=__doc__))
 print('context controls frozen',len(data['ids']),len(columns),flush=True)
if __name__=='__main__':main()
