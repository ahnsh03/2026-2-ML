"""Cal-only secondary severity policy, motivated after the original primary test.

GP/features/calibration and the original policy remain byte-identical. The new
candidate must freeze before the supported-road quality labels are processed.
"""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from quality_estimation.expected_deficit import expected_deficit,far_deficit_with_near_retained
from feature_study import read,save,sha
from review_policy import evaluate

def scores(data,mean_key,covariance_key):
 m=data[mean_key];c=data[covariance_key]
 q=expected_deficit(m[:,0],c[:,0,0],.7)-expected_deficit(m[:,0],c[:,0,0],0.)
 near=expected_deficit(m[:,1],c[:,1,1],.8)-expected_deficit(m[:,1],c[:,1,1],0.)
 far=far_deficit_with_near_retained(m,c,.8,.5)-far_deficit_with_near_retained(m,c,.8,0.)
 return dict(global_deficit=q,near_deficit=near,far_deficit_near_retained=np.maximum(far,0.))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['freeze','evaluate'],required=True);a=p.parse_args();n=a.study
 if a.phase=='freeze':
  a.output.mkdir(parents=True,exist_ok=False)
  assert not (n/'marked_support/evaluation').exists() and not (n/'marked_support/packs').exists()
  with np.load(n/'calibration_v1/selected_oof.npz',allow_pickle=False) as z:d={key:z[key] for key in z.files}
  components=scores(d,'mean','covariance');trials=[]
  for weight in [0.,.25,.5,1.]:
   result=evaluate(components['global_deficit']+.5*components['near_deficit']+weight*components['far_deficit_near_retained'],d)
   trials.append(dict(far_weight=weight,utility=result['utility'],result=result))
  best=max(t['utility'] for t in trials);chosen=min((t for t in trials if t['utility']>=best-.005),key=lambda t:t['far_weight'])
  baseline=evaluate(d['global_probability'],d)
  save(a.output/'frozen.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),primitive_sha256=sha(Path(__file__).parents[3]/'src/quality_estimation/expected_deficit.py'),calibration_sha256=sha(n/'calibration_v1/calibration_frozen.json'),oof_sha256=sha(n/'calibration_v1/selected_oof.npz'),far_weight=chosen['far_weight'],near_weight=.5,global_weight=1.,target_thresholds=[.7,.8,.5],budget=.1,calibration_trials=trials,calibration_baseline=baseline,main_policy_unchanged=True,supported_quality_labels_unopened=True,scope=__doc__))
  print('Expected-deficit candidate frozen',chosen['far_weight'],'cal utility',chosen['utility'],'baseline',baseline['utility'],flush=True);return
 frozen=read(a.output/'frozen.json');assert sha(n/'calibration_v1/calibration_frozen.json')==frozen['calibration_sha256']
 with np.load(n/'marked_support/evaluation/predictions.npz',allow_pickle=False) as z:d={key:z[key] for key in z.files}
 cal=scores(d,'mean','covariance');raw=scores(d,'raw_mean','raw_covariance');score=cal['global_deficit']+.5*cal['near_deficit']+frozen['far_weight']*cal['far_deficit_near_retained']
 candidates=dict(raw_global_deficit=raw['global_deficit'],calibrated_global_deficit=cal['global_deficit'],secondary_frozen_mixture=score,original_frozen_policy=d['priority'],raw_low_mean=-d['raw_mean'][:,0],RF_low_mean=-d['rf_mean'][:,0])
 results={key:evaluate(value,d) for key,value in candidates.items()}
 save(a.output/'supported_results.json',dict(frozen_sha256=sha(a.output/'frozen.json'),predictions_sha256=sha(n/'marked_support/evaluation/predictions.npz'),results=results,scope='Secondary policy frozen after primary results and before supported-road quality scoring. No refit, no replacement of original primary test/policy; descriptor selection only from calibration OOF.'))
 print('Supported expected-deficit utility',{key:r['utility'] for key,r in results.items()},flush=True)
if __name__=='__main__':main()
