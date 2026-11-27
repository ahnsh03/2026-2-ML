"""Compare frozen GP/RF calibration on moving Dev, with no fresh data access."""
from pathlib import Path
import argparse
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from quality_estimation.environment_selection import residual_distribution,calibrated_priority,select_environment_review
from study_data import read,dump,sha,lighting,macro


def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--quality-dir',required=True);a=p.parse_args();S=a.study;root=S/a.quality_dir;choice=read(root/'quality_selection_frozen.json');index=read(S/'protocol/quality_index_frozen.json');rows=[r for r in index['development']['rows'] if r['source_key'].startswith('drive_development_')];calrows=index['calibration']['rows'];sites=np.array([r['site'] for r in rows]);hours=np.array([r['hour'] for r in rows]);records={}
 for kind in ['GP','RF']:
  rec=choice['selected'][kind];path=S/rec['prediction'];z=np.load(path);lookup={s:i for i,s in enumerate(z['ids'])};dev=np.array([lookup[r['sample_id']] for r in rows]);cal=np.array([lookup[r['sample_id']] for r in calrows]);truth=z['Y'];cal=cal[np.isfinite(truth[cal])];m=z['mean'];scale=np.maximum(z['sd'],1e-3) if kind=='GP' else np.ones(len(m));residuals=residual_distribution(truth[cal],m[cal],scale[cal]);prob=calibrated_priority(m[dev],residuals,scale=scale[dev]);lo,hi=np.quantile(residuals,[.05,.95]);lower=np.clip(m[dev]+lo*scale[dev],0,1);upper=np.clip(m[dev]+hi*scale[dev],0,1);y=truth[dev];defined=np.isfinite(y);low=defined&(y<.7);errors=abs(m[dev]-y);covered=(y>=lower)&(y<=upper);brier=(prob-low)**2
  record=dict(defined_rows=int(defined.sum()),calibration_rows=len(cal),macro_mae=macro(errors[defined],sites[defined]),interval90_macro_coverage=macro(covered[defined],sites[defined]),interval90_mean_width=float((upper-lower)[defined].mean()),low_iou_macro_brier=macro(brier[defined],sites[defined]),low_iou_probability_auc=float(roc_auc_score(low[defined],prob[defined])) if len(np.unique(low[defined]))==2 else None,gp_uncertainty_available=kind=='GP',by_lighting={},review=[])
  for l in np.unique(lighting(hours)):
   use=defined&(lighting(hours)==l)
   record['by_lighting'][str(int(l))]=dict(rows=int(use.sum()),mae=float(errors[use].mean()) if use.any() else None,coverage=float(covered[use].mean()) if use.any() else None,brier=float(brier[use].mean()) if use.any() else None)
  if kind=='GP':
   record['predictive_sd_error_spearman']=float(spearmanr(scale[dev][defined],errors[defined]).statistic) if np.ptp(scale[dev][defined])>1e-12 and np.ptp(errors[defined])>1e-12 else None
   record['sd_alone_low_iou_auc']=float(roc_auc_score(low[defined],scale[dev][defined])) if len(np.unique(low[defined]))==2 else None
  for fraction in [.1,.25,3072/7442]:
   selected=select_environment_review(z['ids'][dev],prob,np.zeros(len(dev),int),int(np.ceil(len(dev)*fraction)));chosen=np.isin(np.arange(len(dev)),selected);recall=[float(chosen[(sites==site)&low].mean()) for site in np.unique(sites) if ((sites==site)&low).sum()>=5];record['review'].append(dict(fraction=fraction,selected_rows=len(selected),precision=float(low[selected].mean()),macro_road_recall=float(np.mean(recall)) if recall else None))
  record['prediction_sha256']=sha(path);records[kind]=record
 dump(root/'uncertainty_diagnostics.json',dict(models=records,selected_predictor=choice['policy_predictor'],quality_selection_sha256=sha(root/'quality_selection_frozen.json'),source_sha256=sha(__file__),scope='Same frozen Cal residual recipe and moving Dev membership for both selected point models. No fresh input opened; not used to change the frozen family-selection rule. Lighting and near-deficit remain confounded in this Dev cohort. Predictive-SD/error correlation is distinct from SD/perception-quality correlation.'))
 print('UNCERTAINTY DIAGNOSTICS',a.quality_dir,{kind:{key:r[key] for key in ['macro_mae','interval90_macro_coverage','interval90_mean_width','low_iou_macro_brier']} for kind,r in records.items()},flush=True)
if __name__=='__main__':main()
