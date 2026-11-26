"""Freeze version-paired quality roles before selecting a perception endpoint."""
from pathlib import Path
import argparse,json,hashlib,collections
import numpy as np
from scipy.spatial import cKDTree
from prepare import dump,jsonl

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);a=p.parse_args();S=a.study;N=S.parent/'ml_roi9_drive_20261126'
 rows=[json.loads(s) for s in (S/'protocol/all_sources_v2.jsonl').read_text().splitlines()];cached=[json.loads(s) for s in (S/'protocol/evaluation_cached_v2.jsonl').read_text().splitlines()];lookup={r['sample_id']:r for r in cached}
 def ids(path):return {r['sample_id'] for v in json.loads(path.read_text()).values() for r in v['rows']}
 thin_dev=ids(N/'registry_development_final/index_thinned.json');thin_cal=ids(N/'registry_calibration_final/index_thinned.json')|ids(N/'marked_support/registry/index_thinned.json')
 thin_test=ids(N/'registry_test_final/index_thinned.json')|ids(N/'confirmation/registry/index_thinned.json')|ids(N/'challenge/registry/index_thinned.json')
 roles={'quality_fit':[r for r in rows if r['adaptation_role']=='quality_fit'],
        'development':[r for r in rows if r['adaptation_role']=='development' and (r['sample_id'] in thin_dev or r['source_key'] in ['z05','z10'])],
        'calibration':[r for r in rows if r['adaptation_role']=='calibration' and r['sample_id'] in thin_cal],
        'heldout_diagnostic':[r for r in rows if r['adaptation_role']=='heldout_diagnostic' and r['sample_id'] in thin_test]}
 seen=set();index={};audit={}
 original=json.loads((N/'global_registry_v2/index_full.json').read_text())
 for role,rs in roles.items():
  assert rs and not seen.intersection(r['sample_id'] for r in rs);seen.update(r['sample_id'] for r in rs)
  normalized=[dict(r,simulator_release=original[r['source_key']]['simulator_release']) for r in rs]
  index[role]=dict(simulator_release='26.R1' if role in ['calibration','heldout_diagnostic'] else 'mixed historical, per-row release recorded',rows=normalized)
  jsonl(S/f'protocol/quality_{role}_cached.jsonl',[lookup[r['sample_id']] for r in rs])
 for left,right in [('quality_fit','development'),('quality_fit','calibration'),('quality_fit','heldout_diagnostic'),('calibration','development'),('calibration','heldout_diagnostic'),('development','heldout_diagnostic')]:
  distance=float(cKDTree([r['xy'] for r in roles[left]]).query([r['xy'] for r in roles[right]])[0].min());assert distance>=25;audit[left+'_vs_'+right]=distance
 dump(S/'protocol/quality_index_frozen.json',index)
 dump(S/'protocol/quality_roles_frozen.json',dict(counts={k:len(v) for k,v in roles.items()},minimum_distances_m=audit,
      index_sha256=hashlib.sha256((S/'protocol/quality_index_frozen.json').read_bytes()).hexdigest(),
      quality_fit_membership='Same 2480 frozen g0 IDs; none admitted for perception adaptation',calibration='26R1 only,3m-thinned, excluded all development/heldout neighbours within25m',
      primary_heldout='confirmation three roads; main and condition-shift diagnostics separately',
      historical_static_tests='Retired for spatial overlap; not a g1 heldout result',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 print('QUALITY ROLES',{k:len(v) for k,v in roles.items()},audit,flush=True)
if __name__=='__main__':main()
