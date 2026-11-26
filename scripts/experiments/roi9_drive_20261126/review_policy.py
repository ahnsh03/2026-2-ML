"""Offline review-score search on calibration-road OOF predictions only."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,warnings
import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,ConstantKernel,WhiteKernel
from sklearn.exceptions import ConvergenceWarning

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def priority(global_p,spatial_p,environment_p,weights):
 lam,eta=weights
 return (1-eta)*global_p+eta*environment_p+.5*(spatial_p[:,1]+spatial_p[:,3])+lam*spatial_p[:,2]
def labels(data):
 y=data['truth'];counts=data['counts'];near=counts[:,6:8].sum(1);whole=counts.sum(1)
 npred=near[:,0]+near[:,1];wpred=whole[:,0]+whole[:,1]
 near_p=np.divide(near[:,0],npred,out=np.full(len(near),np.nan),where=npred>0);whole_p=np.divide(whole[:,0],wpred,out=np.full(len(whole),np.nan),where=wpred>0)
 supported=(data['near_gt']>=50)&(data['far_gt']>=50)&np.isfinite(y).all(1)
 return dict(global_low=np.isfinite(y[:,0])&(y[:,0]<.7),near_weak=(data['near_gt']>=50)&np.isfinite(y[:,1])&(y[:,1]<.8),far_only=supported&(y[:,1]>=.8)&(y[:,2]<.5),near_healthy_low_proxy=(data['near_gt']>=50)&(y[:,1]>=.9)&(near_p>=.9)&(whole_p>=.9)&(y[:,0]<.7))
def evaluate(score,data,budget=.1,*,random_seed=None):
 targets=labels(data);roads=data['roads'];ids=data['ids'];selected=[];groups=[]
 for j,road in enumerate(np.unique(roads)):
  ix=np.flatnonzero(roads==road);B=int(np.ceil(len(ix)*budget))
  chosen=(np.random.default_rng(random_seed+j).choice(ix,B,replace=False) if random_seed is not None else ix[np.lexsort((ids[ix],-score[ix]))[:B]])
  selected.extend(chosen.tolist());result=dict(road=str(road),rows=len(ix),budget=B)
  for name,target in targets.items():
   total=int(target[ix].sum());picked=int(target[chosen].sum());result[name]=dict(total=total,selected=picked,recall=picked/total if total else None)
  g=result['global_low'];n=result['near_weak'];f=result['far_only'];normal=result['near_healthy_low_proxy']
  result['utility']=2*(n['recall'] or 0)+(g['recall'] or 0)+.5*(f['recall'] or 0)-.5*normal['selected']/B
  groups.append(result)
 return dict(rows=len(ids),budget=len(selected),actual_budget_fraction=len(selected)/len(ids),utility=float(np.mean([r['utility'] for r in groups])),byroad=groups,counts={name:int(target[np.array(selected)].sum()) for name,target in targets.items()},selected_ids=ids[selected].tolist())
def bayesian_search(objective,seed=23426,trials=8):
 initial=np.array([[0.,0.],[.5,0.],[0.,.5],[1.,1.]])
 grid=np.array([(x,y) for x in np.linspace(0,1,21) for y in np.linspace(0,1,21)])
 records=[];points=initial.tolist();values=[objective(point) for point in points]
 records.extend(dict(weights=x,utility=v,source='shared_initial') for x,v in zip(points,values))
 for _ in range(trials-len(initial)):
  kernel=ConstantKernel(1.,(.01,100))*Matern(length_scale=[.4,.4],length_scale_bounds=(.05,2.),nu=2.5)+WhiteKernel(1e-5,noise_level_bounds='fixed')
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',ConvergenceWarning);gp=GaussianProcessRegressor(kernel=kernel,normalize_y=True,random_state=seed,n_restarts_optimizer=0).fit(points,values)
  mean,sd=gp.predict(grid,return_std=True);gain=mean-max(values)-.001;z=np.divide(gain,sd,out=np.zeros_like(gain),where=sd>1e-12);ei=gain*ndtr(z)+sd*np.exp(-z*z/2)/np.sqrt(2*np.pi)
  for point in points:ei[np.all(np.isclose(grid,point),axis=1)]=-np.inf
  chosen=grid[int(ei.argmax())].tolist();value=objective(chosen);points.append(chosen);values.append(value);records.append(dict(weights=chosen,utility=value,source='expected_improvement',kernel=str(gp.kernel_)))
 rng=np.random.default_rng(seed);rp=initial.tolist();rv=[objective(x) for x in rp];random=[dict(weights=x,utility=v,source='shared_initial') for x,v in zip(rp,rv)]
 for _ in range(trials-len(initial)):
  while True:
   point=grid[int(rng.integers(len(grid)))].tolist()
   if point not in rp:break
  value=objective(point);rp.append(point);random.append(dict(weights=point,utility=value,source='random_search'))
 return dict(BO=records,random=random,evaluations_per_method=trials,shared_initials=4,scope='Policy weights only, on calibration-road OOF utility. GP surrogate does not run online and does not replace qualityGP. Discrete review budgets yield a piecewise-constant objective; no BO advantage is presumed.')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--study',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 frozen=json.loads((a.calibration/'calibration_frozen.json').read_text());assert not any(json.loads(p.read_text())['role']=='test' for p in (a.study/'runs').glob('*/summary.json'))
 with np.load(a.calibration/'selected_oof.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
 gp=data['global_probability'];spatial=data['spatial'];env=data['environment_relative_probability'];cache={}
 def objective(weights):
  key=tuple(float(x) for x in weights)
  if key not in cache:cache[key]=evaluate(priority(gp,spatial,env,key),data)
  return cache[key]['utility']
 search=bayesian_search(objective);anchors=[]
 for lam in [0.,.25,.5,1.]:
  for eta in [0.,1.]:anchors.append(dict(weights=[lam,eta],utility=objective([lam,eta]),source='fixed_anchor'))
 baseline=evaluate(gp,data);allrecords=anchors+search['BO']+search['random'];best=max(r['utility'] for r in allrecords)
 # Favor simpler weights on plateaus; all choices are still development evidence.
 chosen=min([r for r in allrecords if r['utility']>=best-.005],key=lambda r:(sum(r['weights']),r['weights'][1],r['weights'][0]))
 result=cache[tuple(chosen['weights'])];improvements=sum(v['utility']>b['utility'] for v,b in zip(result['byroad'],baseline['byroad']))
 adopt=result['utility']>=baseline['utility']+.01 and improvements>=2
 policy=dict(kind='spatial_environment_mixture' if adopt else 'calibrated_global_probability',weights=chosen['weights'] if adopt else None,near_weight=.5 if adopt else 0.,budget_fraction=.1,spatial_classes=['both_good','near_weak_only','far_weak_only','both_weak'])
 save(a.output/'search.json',dict(search=search,fixed_anchors=anchors,selected=chosen,calibration_baseline=baseline,calibration_candidate=result,adopted_for_prospective_test=adopt,cache_evaluations=len(cache),BO_best=max(r['utility'] for r in search['BO']),random_best=max(r['utility'] for r in search['random'])))
 save(a.output/'policy_frozen.json',dict(version='drive-review-policy-1',created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),calibration_sha256=sha(a.calibration/'calibration_frozen.json'),oof_sha256=sha(a.calibration/'selected_oof.npz'),model_sha256=frozen['model_sha256'],far_auxiliary=frozen.get('far_auxiliary'),policy=policy,scope='Final-test candidate, fitted on calibration-road OOF. Primary targets/thresholds remain unchanged across compared policies. Near-healthy low-Q is a proxy, not operational normality GT.',test_not_collected=True))
 print('policy frozen',policy,'BO/random',max(r['utility'] for r in search['BO']),max(r['utility'] for r in search['random']),flush=True)
if __name__=='__main__':main()
