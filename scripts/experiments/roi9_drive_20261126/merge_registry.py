"""Index reusable raw rows and feature packs without moving immutable sources."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['study','old','data-root','output']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();n=a.study;a.output.mkdir(parents=True,exist_ok=False);index={};packs={};ids=set();pixel_sources={};duplicate_pixels=[]
 sources=[(a.old/'extraction_index_with_acquisition.json',a.old/'packs',None)]
 for role,folder in [('development','packs_development_v2'),('calibration','packs_calibration'),('test','packs_test')]:sources.append((n/('registry_'+role+'_final/index_full.json'),n/folder,None))
 sources.append((n/'challenge/registry/index_full.json',n/'challenge/packs',n))
 if (n/'marked_support/registry/index_full.json').exists():sources.append((n/'marked_support/registry/index_full.json',n/'marked_support/packs',n))
 if (n/'confirmation/registry/index_full.json').exists():sources.append((n/'confirmation/registry/index_full.json',n/'confirmation/packs',n))
 for source,packroot,rebase in sources:
  original=read(source)
  for key,value in original.items():
   if key in index:raise ValueError('duplicate source key')
   rows=[]
   for row in value['rows']:
    r=dict(row)
    if rebase:
     for field in ['image','semantic']:r[field]=str((rebase/r[field]).relative_to(a.data_root))
    if r['sample_id'] in ids:raise ValueError('duplicate sample ID')
    ids.add(r['sample_id']);rows.append(r)
   path=packroot/(key+'.npz');receipt=read(path.with_suffix('.json'))
   assert sha(path)==receipt['sha256'] and sha(packroot/'schema.json')==receipt['schema_sha256']
   with np.load(path,allow_pickle=False) as z:
    assert z['ids'].tolist()==[r['sample_id'] for r in rows]
    for sid,digest in zip(z['ids'],z['pixel_sha256']):
     if str(digest) in pixel_sources:duplicate_pixels.append(dict(sample_id=str(sid),previous_id=pixel_sources[str(digest)]))
     else:pixel_sources[str(digest)]=str(sid)
   index[key]=dict(value,rows=rows)
   packs[key]=dict(artifact=str(path.relative_to(a.data_root)),sha256=receipt['sha256'],schema=str((packroot/'schema.json').relative_to(a.data_root)),schema_sha256=receipt['schema_sha256'],original_index=str(source.relative_to(a.data_root)),original_index_sha256=sha(source),perception_checkpoint_sha256=receipt['checkpoint_sha256'],rows=len(rows))
 (a.output/'index_full.json').write_text(json.dumps(index,indent=2)+'\n');(a.output/'pack_artifacts.json').write_text(json.dumps(packs,indent=2)+'\n')
 excluded=[];excluded_complete=[];canonical_runs={str(sid).rsplit('_f',1)[0] for sid in ids}
 for root in [n,n/'challenge',n/'marked_support',n/'confirmation']:
  if not (root/'runs').exists():continue
  for run in sorted((root/'runs').iterdir()):
   if run.is_dir() and not (run/'_SUCCESS').exists():excluded.append(str(run.relative_to(n)))
   elif run.is_dir() and run.name not in canonical_runs:excluded_complete.append(str(run.relative_to(n)))
 result=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),packs=len(packs),unique_ids=len(ids),unique_decoded_rgb=len(pixel_sources),duplicate_decoded_pixels=duplicate_pixels,excluded_partial_runs=excluded,excluded_complete_pilot_runs=excluded_complete,index_sha256=sha(a.output/'index_full.json'),pack_artifacts_sha256=sha(a.output/'pack_artifacts.json'),scope='Canonical complete rows only. No raw/source files moved or deleted; partial/pilot raw captures remain quarantined. Thinned scientific cohort indices remain separate. New nested followup paths rebased to the supplied common data root.')
 (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
