"""Uncertainty ablation for the secondary deficit score; no fitting/selection."""
from pathlib import Path
import argparse,numpy as np
from feature_study import read,save,sha
from review_policy import evaluate
from expected_deficit_study import scores

def point_score(mean,far_weight):
 m=np.clip(mean,0,1)
 return np.maximum(.7-m[:,0],0)+.5*np.maximum(.8-m[:,1],0)+far_weight*(m[:,1]>=.8)*np.maximum(.5-m[:,2],0)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();n=a.study;frozen=read(n/'expected_deficit_secondary/frozen.json');protocol=read(n/'expected_deficit_ablation_before_supported_scores.json')
 assert protocol['created_utc']<read(n/'marked_support/evaluation/results.json')['created_utc']
 assert sha(Path(__file__).with_name('expected_deficit_study.py'))==frozen['source_sha256']
 with np.load(n/'marked_support/evaluation/predictions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 c=scores(d,'mean','covariance');weight=frozen['far_weight'];mixture=c['global_deficit']+.5*c['near_deficit']+weight*c['far_deficit_near_retained']
 candidates=dict(GP_posterior_deficit=mixture,GP_calibrated_means_only=point_score(d['mean'],weight),GP_raw_means_only=point_score(d['raw_mean'],weight),RF_means_only=point_score(d['rf_mean'],weight))
 save(a.output,dict(protocol_sha256=sha(n/'expected_deficit_ablation_before_supported_scores.json'),candidate_sha256=sha(n/'expected_deficit_secondary/frozen.json'),results={key:evaluate(score,d) for key,score in candidates.items()},scope='Fixed identical task weights compare posterior integration against mean-only ablation and matched RF multi-target means. No claim of uncertainty benefit unless this ablation supports it; no model/policy selection.'))
 print('Expected-deficit ablation saved',a.output,flush=True)
if __name__=='__main__':main()
