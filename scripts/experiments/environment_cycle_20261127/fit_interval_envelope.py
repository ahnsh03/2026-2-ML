"""Compare pooled/road-envelope intervals without changing means or review scores."""
from pathlib import Path
import argparse
import numpy as np
from study_data import read,dump,sha,lighting,macro


def fit_bounds(residuals,roads,method):
 pooled=np.quantile(residuals,[.05,.95]);groups=[residuals[roads==g] for g in np.unique(roads) if (roads==g).sum()>=30]
 if method=='pooled' or len(groups)<2:return pooled
 tails=np.array([np.quantile(values,[.05,.95]) for values in groups]);return np.array([tails[:,0].min(),tails[:,1].max()])


def metrics(y,mean,scale,groups,bounds):
 lo=np.clip(mean+bounds[0]*scale,0,1);hi=np.clip(mean+bounds[1]*scale,0,1);covered=(y>=lo)&(y<=hi);width=hi-lo;score=width+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0)
 return dict(coverage=macro(covered,groups),width=macro(width,groups),interval_score=macro(score,groups),rows=len(y))


def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--quality-dir',required=True);a=p.parse_args();S=a.study;root=S/a.quality_dir;choice=read(root/'quality_selection_frozen.json');index=read(S/'protocol/quality_index_frozen.json');calrows=index['calibration']['rows'];devrows=[r for r in index['development']['rows'] if r['source_key'].startswith('drive_development_')];result={}
 for kind in ['GP','RF']:
  rec=choice['selected'][kind];z=np.load(S/rec['prediction']);lookup={s:i for i,s in enumerate(z['ids'])};cal=np.array([lookup[r['sample_id']] for r in calrows]);dev=np.array([lookup[r['sample_id']] for r in devrows]);roads=np.array([r['site'] for r in calrows]);droads=np.array([r['site'] for r in devrows]);hours=np.array([r['hour'] for r in devrows]);q=z['Y'];m=z['mean'];scale=np.maximum(z['sd'],1e-3) if kind=='GP' else np.ones(len(q));valid=np.isfinite(q[cal]);cal=cal[valid];roads=roads[valid];valid=np.isfinite(q[dev]);dev=dev[valid];droads=droads[valid];hours=hours[valid];trials={}
  if len(np.unique(roads))<3:raise ValueError('three calibration roads required for interval comparison')
  for method in ['pooled','road_envelope']:
   cv=[]
   for road in np.unique(roads):
    train=cal[roads!=road];test=cal[roads==road];bounds=fit_bounds((q[train]-m[train])/scale[train],roads[roads!=road],method);cv.append(metrics(q[test],m[test],scale[test],roads[roads==road],bounds))
   bounds=fit_bounds((q[cal]-m[cal])/scale[cal],roads,method);development=metrics(q[dev],m[dev],scale[dev],droads,bounds);by_light={str(int(l)):metrics(q[dev][lighting(hours)==l],m[dev][lighting(hours)==l],scale[dev][lighting(hours)==l],droads[lighting(hours)==l],bounds) for l in np.unique(lighting(hours))};cross={key:float(np.mean([r[key] for r in cv])) for key in ['coverage','width','interval_score']};trials[method]=dict(cal_leave_road_out=cross,development=development,development_by_lighting=by_light,bounds=bounds.tolist(),objective=(cross['interval_score']+development['interval_score'])/2)
  pooled,envelope=trials['pooled'],trials['road_envelope'];adopt=envelope['objective']<=pooled['objective']-.005 and envelope['development']['coverage']>=pooled['development']['coverage'];selected='road_envelope' if adopt else 'pooled';result[kind]=dict(selected_method=selected,interval_residual_bounds=trials[selected]['bounds'],trials=trials,calibration_rows=len(cal),calibration_roads=len(np.unique(roads)),defined_primary_development_rows=len(dev),model_sha256=rec['model_sha256'])
 dump(root/'interval_calibration_study.json',dict(models=result,quality_selection_sha256=sha(root/'quality_selection_frozen.json'),source_sha256=sha(__file__),target_coverage=.9,selection_rule='Two fixed recipes. Adopt road envelope only for mean Cal-leave-road-out/primary-Dev90% interval-score improvement>=0.005 and nondecreasing Dev coverage. Width is penalized by the proper interval score, not optimized by coverage alone.',scope='Cal road strata only; no road ID or GT at runtime. Fresh remains unopened. Signed residual CDF review scores and point means are unchanged. f0 study is diagnostic and does not replace its frozen bank/bundle. New g1 export binds the selected bounds. Empirical validation, not a finite-sample or conditional-coverage guarantee.'))
 print('INTERVAL CALIBRATION',a.quality_dir,{kind:dict(method=r['selected_method'],dev=r['trials'][r['selected_method']]['development']) for kind,r in result.items()},flush=True)
if __name__=='__main__':main()
