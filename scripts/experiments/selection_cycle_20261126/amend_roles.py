"""Preserve v1 and correct an ambiguous calibration prefix before any f1 heldout evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,collections
import numpy as np
from scipy.spatial import cKDTree
from prepare import dump,jsonl

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);a=p.parse_args();S=a.study
 z=np.load(S/'source_packs.npz');d={k:z[k] for k in z.files};rows=[json.loads(s) for s in (S/'protocol/all_sources.jsonl').read_text().splitlines()];old=d['role'].copy();role=old.astype('<U48')
 for i,r in enumerate(rows):
  if r['source_key'].startswith('drive_calibration_shift_'):role[i]='heldout_diagnostic'
  elif r['source_key'] in ['z04','z07','z13']:role[i]='historical_spatial_overlap'
 # Calibration labels cannot come from near-neighbours of development/heldout roads.
 reserved=np.isin(role,['development','heldout_diagnostic']);distance=cKDTree(d['xy'][reserved]).query(d['xy'])[0]
 role[(role=='calibration')&(distance<25)]='calibration_buffer_excluded'
 assert np.array_equal(old=='candidate_pool',role=='candidate_pool') and np.array_equal(old=='quality_fit',role=='quality_fit') and np.array_equal(old=='development',role=='development')
 for r,value in zip(rows,role):r['adaptation_role']=str(value)
 d['role']=role;np.savez_compressed(S/'source_packs_v2.npz',**d);jsonl(S/'protocol/all_sources_v2.jsonl',rows)
 ids={r['sample_id']:r for r in rows}
 cached=[json.loads(s) for s in (S/'protocol/evaluation_cached.jsonl').read_text().splitlines()]
 for r in cached:r['adaptation_role']=ids[r['sample_id']]['adaptation_role']
 jsonl(S/'protocol/evaluation_cached_v2.jsonl',cached)
 audit={}
 for left,right in [('quality_fit','development'),('quality_fit','calibration'),('quality_fit','heldout_diagnostic'),('calibration','development'),('calibration','heldout_diagnostic'),('development','heldout_diagnostic')]:
  a_=role==left;b=role==right;minimum=float(cKDTree(d['xy'][a_]).query(d['xy'][b])[0].min());audit[left+'_vs_'+right]=minimum
  if minimum<25:raise ValueError('amended quality role spatial embargo failed: '+left+' '+right)
 dump(S/'protocol/roles_v2_frozen.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),original_roles=dict(collections.Counter(old)),roles=dict(collections.Counter(role)),
       reason='drive_calibration_shift prefix was swallowed by drive_calibration. Retire historical static tests near quality-fit positions, and embargo calibration against dev/heldout.',
       candidate_and_quality_fit_and_development_membership_unchanged=True,active_training_admissions_unchanged=True,
       v1_inputs_preserved=True,minimum_role_distances_m=audit,
       downstream_invalidated=['v1 relative-reference policy calibration only; no added perception arm had used it'],
       valid_prior_results=['e-means candidate-only fit and metadata study','common-threshold BO (dev only)','f0 auxiliary GP feasibility (fit/dev only)','current perception training admission manifests'],
       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 print('AMENDED',collections.Counter(role),audit,flush=True)
if __name__=='__main__':main()
