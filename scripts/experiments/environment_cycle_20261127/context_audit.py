"""Expose environment/deficit confounding and compare context-only Q baselines."""
from pathlib import Path
import argparse,json
import numpy as np
from study_data import read,dump,sha,load_sources,memberships,labels,lighting,macro

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();S=a.study;data,_,rows=load_sources(a.parent);m=memberships(S,data['ids']);q,h,n=labels(data);sim=data['weather'].astype(int)*3+lighting(data['hour']);dev=m['development'];primary=dev[np.array([rows[i]['source_key'].startswith('drive_development_') for i in dev])];fit=m['quality_fit'];report={}
 for name,ix in [('fit',fit),('calibration',m['calibration']),('primary_dev',primary),('historical_dev',np.setdiff1d(dev,primary))]:
  part={}
  for l in [0,1,2]:
   ids=ix[lighting(data['hour'][ix])==l];b=ids[n[ids]];near_gt=(data['counts'][ids,6:8,0]+data['counts'][ids,6:8,2]).sum(1);whole_gt=(data['counts'][ids,:,0]+data['counts'][ids,:,2]).sum(1)
   part[str(l)]=dict(rows=len(ids),near_deficit=int(h[ids].sum()),near_retained=int(n[ids].sum()),zero_lane_gt=int((whole_gt==0).sum()),retained_q_quantiles=np.quantile(q[b],[.1,.5,.9]).tolist() if len(b) else None)
  report[name]=part
 baseline={}
 for name,ix in [('primary',primary),('historical',np.setdiff1d(dev,primary))]:
  mean=float(np.mean(q[fit]));group_mean={int(g):float((q[fit][sim[fit]==g].sum()+5*mean)/((sim[fit]==g).sum()+5)) for g in np.unique(sim[fit])};pred=np.array([group_mean.get(int(g),mean) for g in sim[ix]]);baseline[name]=dict(context_only_oracle_mae=macro(abs(pred-q[ix]),data['site'][ix]),constant_mean_mae=macro(abs(mean-q[ix]),data['site'][ix]))
  for kind in ['GP','RF']:
   rec=read(S/'f0_quality/quality_selection_frozen.json')['selected'][kind];z=np.load(S/rec['prediction']);pred=z['mean'][ix];err=abs(pred-q[ix]);groups={str(int(g)):float(np.mean(err[sim[ix]==g])) for g in np.unique(sim[ix]) if (sim[ix]==g).sum()>=10};baseline[name][kind]=dict(macro_road_mae=macro(err,data['site'][ix]),per_environment_mae=groups)
 report.update(regression_baselines=baseline,scope='Day/night near-deficit confounding is measured explicitly. Supported near-retained precision/recall is a proxy, not visibility-aware operating-normality truth. Context-only baseline uses known simulator tags, not a learned weather-classification claim.',source_sha256=sha(__file__))
 dump(S/'reports/environment_context_audit.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
