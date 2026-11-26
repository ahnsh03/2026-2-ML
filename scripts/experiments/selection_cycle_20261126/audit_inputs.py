"""Audit frozen inputs and regenerated original replay/validation masks without copying raw data."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,hashlib
import cv2,numpy as np
from common.bridge import sha256_file
from quality_estimation.image_io import decoded_pixel_sha256
from prepare import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--original-data',type=Path,required=True);p.add_argument('--original-hashes',action='store_true');p.add_argument('--extra-original-manifest',type=Path);p.add_argument('--report-name',default='input_and_mask_audit.json');a=p.parse_args();S=a.study;D=a.original_data;cv2.setNumThreads(1)
 E=S.parent;N=E/'ml_roi9_drive_20261126';art=json.loads((N/'global_registry_v2/pack_artifacts.json').read_text());verified=[]
 for key,v in art.items():
  for field,h in [('artifact','sha256'),('schema','schema_sha256'),('original_index','original_index_sha256')]:
   if sha256_file(E/v[field])!=v[h]:raise ValueError('source artifact changed: '+key+':'+field)
  verified.append(key)
 version=D/'dataset_versions/twinlite_morai_v6_road_marking_actor_negative';source={}
 expected={'train':'277f9161f39a0df0c99aaf9f23b07043b8d62af368bda1d425a5ea52c7ac8d85','val':'4add24d993fabacb4a99f9d36caaee7c7d4ffc7c391e33d4c5d087a4d64092d4'}
 for split in ['train','val']:
  path=version/(split+'.jsonl')
  if sha256_file(path)!=expected[split]:raise ValueError('original f0 exposure manifest changed')
  for row in map(json.loads,path.read_text().splitlines()):source[row['sample_id']]=dict(row,source_split=split)
 cached=[json.loads(s) for s in (S/'protocol/cached_sources.jsonl').read_text().splitlines()]
 if a.extra_original_manifest:cached += [json.loads(s) for s in a.extra_original_manifest.read_text().splitlines()]
 original=[r for r in cached if r['adaptation_role'] in ['replay','retention','replay_replacement']]
 if len({r['sample_id'] for r in original})!=len(original):raise ValueError('duplicate original audit membership')
 def compare(row):
  source_row=source[row['sample_id']];target=cv2.imread(str(S/'cache'/row['cache_relative']/'targets.png'),cv2.IMREAD_UNCHANGED)
  for j,name in enumerate(['drivable','road_marking']):
   old=cv2.imread(str(D/source_row['targets'][name]),cv2.IMREAD_UNCHANGED)
   if old is None or not np.array_equal(old,target[:,:,j]):raise ValueError('regenerated original target differs: '+row['sample_id']+':'+name)
  valid=cv2.imread(str(D/source_row['valid_mask']),cv2.IMREAD_UNCHANGED)
  if not np.array_equal(valid.astype(bool),target[:,:,2].astype(bool)):raise ValueError('padding mask differs')
  return row['sample_id']
 with ThreadPoolExecutor(4) as pool:matched=list(pool.map(compare,original))
 report=dict(verified_pack_keys=verified,verified_pack_count=len(verified),original_manifest_sha256=expected,mask_parity_samples=len(matched),
             two_head_targets_bitwise_equal=True,geometric_padding_equal=True,source_files_copied=False)
 dump(S/'reports'/a.report_name,report);print('MASK PARITY',len(matched),flush=True)
 if a.original_hashes:
  all_rows=list(source.values());path=S/'reports/original_rgb_exposure.jsonl'
  def identify(row):
   file=D/row['image'];data=file.read_bytes();bgr=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_UNCHANGED)[:,:,:3];rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
   return dict(sample_id=row['sample_id'],split=row['source_split'],rgb_sha256=hashlib.sha256(data).hexdigest(),pixel_sha256=decoded_pixel_sha256(rgb))
  if not path.exists():
   with ThreadPoolExecutor(4) as pool,path.open('x') as stream:
    for j,r in enumerate(pool.map(identify,all_rows)):
     stream.write(json.dumps(r)+'\n')
     if j%1000==0:stream.flush();print('original RGB audit',j+1,'/',len(all_rows),flush=True)
  exposure=[json.loads(s) for s in path.read_text().splitlines()];assert len(exposure)==len(all_rows)
  before={r['pixel_sha256'] for r in exposure};new=[r for r in cached if r['adaptation_role']=='candidate_pool'];overlap=[r['sample_id'] for r in new if r['pixel_sha256'] in before]
  dump(S/'reports/original_exposure_audit.json',dict(original_rows=len(exposure),original_unique_pixels=len(before),candidate_rows=len(new),candidate_pixel_overlap=overlap,exposure_sha256=sha256_file(path)))
  if overlap:raise ValueError('candidate was already exposed to f0')
if __name__=='__main__':main()
