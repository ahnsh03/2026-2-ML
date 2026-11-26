"""Bind immutable admissions to their actual frozen g0 vectors/model and RGB-only clusters."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.cluster_review import review_scores,select_review
from prepare import dump

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);a=p.parse_args();S=a.study;z=np.load(S/'source_packs_v2.npz');d={k:z[k] for k in z.files};schema=json.loads((S/'schema.json').read_text());policy=json.loads((S/'protocol/policy_frozen.json').read_text())
 pool=np.flatnonzero(d['role']=='candidate_pool');dev=np.flatnonzero(d['role']=='development');ix=np.concatenate([pool,dev]);modelpath=S.parent/'ml_roi9_gp_20261125/frozen_bundle_final/quality_spatial_gp.npz';model=SpatialQualityGP.load(modelpath);x=feature_matrix(d,schema,model.feature_names_.tolist());delta_m=delta_c=0.
 for start in range(0,len(ix),256):
  use=ix[start:start+256];mean,cov=model.predict(x[use]);delta_m=max(delta_m,float(abs(mean-d['g0_mean'][use]).max()));delta_c=max(delta_c,float(abs(cov-d['g0_covariance'][use]).max()))
 assert delta_m<1e-10 and delta_c<1e-10
 cluster=policy['cluster'];archive=np.load(S/cluster['path']);labels=archive['assignment'];v=policy['selection_parameters'];score=review_scores(d['g0_mean'][pool,0],d['g0_covariance'][pool,0,0],uncertainty_weight=v['uncertainty_weight'])
 selections={'Random':np.random.default_rng(261126).choice(pool,3072,replace=False),
             'QualityRank':pool[np.lexsort((d['ids'][pool],-review_scores(d['g0_mean'][pool,0],d['g0_covariance'][pool,0,0])))[:3072]],
             'ClusterBO':pool[select_review(d['ids'][pool],score,labels[pool],3072,quota_exponent=v['quota_exponent'],audit_fraction=v['audit_fraction'])]}
 admissions={}
 for arm,indices in selections.items():
  path=S/f'protocol/{arm}_sources.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()];assert [r['sample_id'] for r in rows]==d['ids'][indices].tolist()
  admissions[arm]=dict(rows=len(rows),sources_sha256=sha(path),cached_sha256=sha(S/f'protocol/{arm}_cached.jsonl'),reproduced_from_features_without_candidate_gt=True)
 dump(S/'protocol/selection_lineage_receipt.json',dict(g0_model_sha256=sha(modelpath),g0_perception_sha256=str(model.checkpoint_sha256_),g0_feature_order=model.feature_names_.tolist(),
      source_packs_v1_sha256=sha(S/'source_packs.npz'),source_packs_v2_sha256=sha(S/'source_packs_v2.npz'),candidate_and_dev_prediction_rows=len(ix),
      prediction_max_difference=delta_m,covariance_max_difference=delta_c,cluster_archive_sha256=sha(S/cluster['path']),policy_sha256=sha(S/'protocol/policy_frozen.json'),admissions=admissions,
      uncertainty_tie_count=len(pool)-len(np.unique(d['g0_covariance'][pool,0,0])),
      scope='Extends the pre-training immutable policy receipt with recomputed numerical lineage. All three admission ID orders reproduce exactly; no candidate GT inspected for selection.'))
 print('ADMISSIONS REPRODUCED',delta_m,delta_c,flush=True)
if __name__=='__main__':main()
