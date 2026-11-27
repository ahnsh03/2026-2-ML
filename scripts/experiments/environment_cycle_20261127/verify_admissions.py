"""Reproduce admissions from X/models/Cal residuals with no candidate GT loaded."""
from pathlib import Path
import argparse,json
import numpy as np
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP
from quality_estimation.portable_forest import PortableForest
from quality_estimation.environment_selection import calibrated_priority,select_environment_review
from study_data import read,dump,sha,lighting

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();S=a.study;P=a.parent;z=np.load(P/'source_packs_v2.npz');fields=['ids','role','weather','hour','rgb','output','relations','uncertainty','morphology'];data={key:z[key] for key in fields};assert 'y' not in data and 'counts' not in data;pool=np.flatnonzero(data['role']=='candidate_pool');ids=data['ids'][pool];sim=data['weather'][pool].astype(int)*3+lighting(data['hour'][pool]);policy=read(S/'protocol/selection_policy_frozen.json');quality=read(S/'f0_quality/quality_selection_frozen.json');schema=read(P/'schema.json');report={}
 for kind in ['GP','RF']:
  rec=quality['selected'][kind];model=SpatialQualityGP.load(S/rec['model']) if kind=='GP' else PortableForest.load(S/rec['model']);x=feature_matrix(data,schema,rec['columns'])[pool];means=[];scales=[]
  for start in range(0,len(x),256):
   if kind=='GP':m,c=model.predict(x[start:start+256]);means.append(np.clip(m[:,0],0,1));scales.append(np.sqrt(c[:,0,0]))
   else:means.append(model.predict(x[start:start+256]));scales.append(np.ones(min(256,len(x)-start)))
  mean=np.concatenate(means);scale=np.maximum(np.concatenate(scales),1e-3);pred=np.load(S/rec['prediction']);difference=float(np.max(abs(mean-pred['mean'][pool])));assert difference<1e-10
  score=calibrated_priority(mean,np.asarray(policy['quality_models'][kind]['residuals']),scale=scale);ix=select_environment_review(ids,score,sim,3072);arm=kind+'GlobalFull';rows=[json.loads(s) for s in (S/f'protocol/{arm}_cached.jsonl').read_text().splitlines()];assert [r['sample_id'] for r in rows]==ids[ix].tolist();report[arm]=dict(reproduced_ordered_ids=True,rows=len(ix),prediction_max_difference=difference,manifest_sha256=sha(S/f'protocol/{arm}_cached.jsonl'))
 score=np.random.default_rng(271127).random(len(pool));ix=select_environment_review(ids,score,sim,3072,balanced_floor=1.,audit_fraction=0.);rows=[json.loads(s) for s in (S/'protocol/TagBalancedFull_cached.jsonl').read_text().splitlines()];assert [r['sample_id'] for r in rows]==ids[ix].tolist();report['TagBalancedFull']=dict(reproduced_ordered_ids=True,rows=len(ix))
 actual=[json.loads(s)['sample_id'] for s in (S/'protocol/RandomFull_cached.jsonl').read_text().splitlines()];parent=[json.loads(s)['sample_id'] for s in (P/'protocol/Random_cached.jsonl').read_text().splitlines()];assert actual==parent;report['RandomFull']=dict(ordered_ids_preserved_from_frozen_parent_random_bank=True,rows=len(actual),parent_manifest_sha256=sha(P/'protocol/Random_cached.jsonl'))
 dump(S/'reports/selection_lineage_no_candidate_gt.json',dict(loaded_fields=fields,candidate_gt_loaded=False,candidate_semantic_images_read=False,quality_models=report,source_sha256=sha(__file__),policy_sha256=sha(S/'protocol/selection_policy_frozen.json')));print('SELECTION REPRODUCED WITHOUT CANDIDATE GT',flush=True)
if __name__=='__main__':main()
