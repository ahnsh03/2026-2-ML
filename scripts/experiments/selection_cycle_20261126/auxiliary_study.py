"""Dev-only FP/FN-aware GP feasibility, with frozen g0 membership and feature ablations."""
from pathlib import Path
import argparse,json,time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.diagnostic_targets import auxiliary_targets,FOUR_TARGET_NAMES
from prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);a=p.parse_args();S=a.study;out=S/'auxiliary_study';out.mkdir(exist_ok=True)
 schema=json.loads((S/'schema.json').read_text());z=np.load(S/'source_packs.npz');d={k:z[k] for k in z.files};Y=auxiliary_targets(d['counts'],d['y'])
 recipe=json.loads((S.parent/'ml_roi9_gp_20261125/frozen_bundle_final/feature_recipe.json').read_text());base=recipe['tasks']['quality_spatial_gp']['ordered_features']
 extra=['near_left.lane_drivable_disagreement','near_right.lane_drivable_disagreement','global.lane_drivable_disagreement','near.soft_outside','near_left.internal_gap','near_right.internal_gap']
 fit=(d['role']=='quality_fit')&np.isfinite(Y).all(1);dev=d['role']=='development';assert fit.sum()==2480
 records=[]
 for name,columns in [('base28',base),('fp34',base+extra)]:
  x=feature_matrix(d,schema,columns)
  for kind in ['GP','RF']:
   t=time.monotonic()
   if kind=='GP':
    model=SpatialQualityGP(**recipe['tasks']['quality_spatial_gp']['parameters']).fit(x[fit],Y[fit]);model.feature_names_=np.array(columns);model.target_names_=np.array(FOUR_TARGET_NAMES);model.save(out/f'{name}_GP.npz');mean,cov=model.predict(x[dev]);np.savez_compressed(out/f'{name}_GP_development.npz',ids=d['ids'][dev],mean=mean,covariance=cov,Y=Y[dev],site=d['site'][dev])
   else:
    model=RandomForestRegressor(n_estimators=300,min_samples_leaf=3,max_features=.7,random_state=261126,n_jobs=2).fit(x[fit],Y[fit]);mean=model.predict(x[dev]);cov=None
   valid=np.isfinite(Y[dev]);errors=abs(np.clip(mean,0,1)-Y[dev]);roads=d['site'][dev]
   mae=[float(np.mean([errors[(roads==road)&valid[:,j],j].mean() for road in np.unique(roads) if ((roads==road)&valid[:,j]).sum()])) for j in range(4)]
   rec=dict(view=name,kind=kind,columns=columns,train_rows=int(fit.sum()),development_macro_mae=mae,seconds=time.monotonic()-t,role='Feasibility only; no perception finetuning selection or final-test labels used')
   records.append(rec);print(rec['view'],kind,'MAE',mae,flush=True)
 dump(out/'development_results.json',dict(records=records,targets=list(FOUR_TARGET_NAMES),thresholds={'near_f1_diagnostic':.8,'absolute_quality':.7},scope='Near F1 is an offline learned precision/recall auxiliary; it does not label night driving normality or operational safety.'))
if __name__=='__main__':main()
