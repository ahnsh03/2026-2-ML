"""Hash-audit fresh captures and freeze geometry-thinned, test-only membership."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from study_data import read,dump,sha,load_sources

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();S=a.study;C=S/'confirmation';out=C/'registry';out.mkdir(exist_ok=False);plans=[C/'plans/frozen.json',C/'plans/transition_frozen.json'];accepted={sha(p) for p in plans};good=[];allrows=[];checks=[];restores=[]
 for success in sorted((C/'runs').glob('*/_SUCCESS')):
  root=success.parent;summary=read(root/'summary.json')
  if summary['plan_sha256'] not in accepted:continue
  good.append(summary);values=read(root/'index.json');rows=next(iter(values.values()))['rows'];assert len(rows)==summary['frames'];manifest=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()];by_id={r['sample_id']:r for r in manifest if r.get('accepted')};assert len(by_id)==len(rows)
  for row in rows:
   raw=by_id[row['sample_id']];assert raw['quality_role']=='fresh_confirmation_test'
   for channel,file in raw['files'].items():checks.append((C/file['path'],file['sha256']))
   r=dict(row,image='e/'+S.name+'/'+row['image'],semantic='e/'+S.name+'/'+row['semantic'],adaptation_role='heldout_diagnostic',source_role='after_f0_training',source_key='fresh_confirmation_'+row['site'],parent_session_id=summary['run_id'],simulator_release='26.R1',original_hw=[720,1280],original_collection_role='fresh_confirmation_test',collection_plan_sha256=summary['plan_sha256']);allrows.append(r)
 def verify(item):
  p,digest=item
  if sha(p)!=digest:raise ValueError('fresh capture bytes changed: '+str(p))
 with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(verify,checks))
 # Within each independent condition/drive: retain first then >=3m XY travel.
 thin=[]
 for summary in good:
  rows=[r for r in allrows if r['parent_session_id']==summary['run_id']];last=None
  for row in rows:
   xy=np.array(row['xy'])
   if last is None or np.linalg.norm(xy-last)>=3:thin.append(row);last=xy
 data,_,_=load_sources(a.parent);distance=cKDTree(data['xy']).query(np.array([r['xy'] for r in allrows]))[0];assert distance.min()>=25
 assert len({r['sample_id'] for r in thin})==len(thin)
 for prefix in sorted({r['run_id'].split('_env28_')[0] for r in good}):
  result=read(C/(prefix+'_completion.json'));assert result['restoration']['verified'];restores.append(prefix)
 index={'fresh_confirmation':{'simulator_release':'26.R1','rows':thin}};dump(out/'index_frozen.json',index);dump(out/'index_all.json',{'fresh_confirmation':{'simulator_release':'26.R1','rows':allrows}})
 dump(out/'alignment/index_full.json',{run:{'simulator_release':'26.R1','rows':[r for r in allrows if r['parent_session_id']==run]} for run in sorted({r['parent_session_id'] for r in allrows})})
 dump(out/'audit.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),conditions=len(good),raw_frames=len(allrows),thinned_frames=len(thin),verified_4channel_files=len(checks),minimum_old_source_distance_m=float(distance.min()),maximum_cte_m=max(r['maximum_cte_m'] for r in good),maximum_speed_mps=max(r['maximum_speed_mps'] for r in good),all_successful_epochs_restoration_verified=True,source_sha256=sha(__file__),index_sha256=sha(out/'index_frozen.json'),plans_sha256=sorted(accepted),scope='Same-map locations buffered from registered tuning/adaptation captures; original-v6-training-route exclusion is not established.3m XY thinning before target/prediction inspection. No exposure timestamps; content alignment still separate.'))
 print('CONFIRMATION REGISTERED',len(allrows),len(thin),flush=True)
if __name__=='__main__':main()
