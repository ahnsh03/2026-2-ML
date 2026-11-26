"""Audit completed moving runs and register reusable full/thinned candidate rows."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from data_collection.mask_baker import OFFICIAL_PALETTE_RGB

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['study','plan','prior-index','prior-packs','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--roles',nargs='+',required=True);p.add_argument('--spacing',type=float,default=3)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 plan=json.loads(a.plan.read_text());routes={r['route_id']:r for r in plan['routes']};old=json.loads(a.prior_index.read_text())
 prior=cKDTree([r['xy'] for v in old.values() for r in v['rows']]);prior_ids={r['sample_id'] for v in old.values() for r in v['rows']}
 old_pixels=set()
 for key in old:
  with np.load(a.prior_packs/(key+'.npz'),allow_pickle=False) as z:old_pixels.update(z['pixel_sha256'].tolist())
 palette={r*65536+g*256+b for r,g,b in OFFICIAL_PALETTE_RGB.values()};blue=OFFICIAL_PALETTE_RGB['blue_lane'];bluecode=blue[0]*65536+blue[1]*256+blue[2]
 all_index={};thin_index={};selected={};seenids=set();pixels=set();rejected=[];inventory=[];cases=[]
 for success in sorted((a.study/'runs').glob('*/_SUCCESS')):
  run=success.parent
  if not run.name.startswith(('main_v1_','main_v2_','udp_main_')):continue
  summary=json.loads((run/'summary.json').read_text())
  if summary['role'] not in a.roles:continue
  assert summary['plan_sha256']==sha(a.plan)
  key=(summary['route_id'],summary['weather'],summary['hour'])
  if key in selected:raise ValueError('more than one completed run for route/condition; choose explicitly')
  selected[key]=run.name;source=json.loads((run/'design.json').read_text());assert sha(run/'collector_source.py')==source['source_sha256']
  original=next(iter(json.loads((run/'index.json').read_text()).values()))['rows'];lookup={r['sample_id']:r for r in original};eligible=[]
  for line in (run/'manifest.jsonl').read_text().splitlines():
   m=json.loads(line)
   if not m['accepted']:continue
   sid=m['sample_id'];r=dict(lookup[sid]);assert sid not in prior_ids and sid not in seenids;seenids.add(sid)
   # Original per-run indexes used the container mount name; raw records stay immutable.
   r['image']=a.study.name+'/'+m['files']['Intensity']['path']
   r['semantic']=a.study.name+'/'+m['files']['Semantic']['path']
   for channel in ['Intensity','Semantic','Depth','Instance']:
    item=m['files'][channel];file=a.study/item['path'];assert sha(file)==item['sha256']
    with Image.open(file) as image:assert image.size==(1280,720);image.verify()
   with Image.open(a.study/m['files']['Intensity']['path']) as im:rgb=np.asarray(im.convert('RGB'))
   pixelsha=hashlib.sha256(np.asarray(rgb.shape,dtype=np.int64).tobytes()+np.ascontiguousarray(rgb).tobytes()).hexdigest()
   r['expected_pixel_sha256']=pixelsha
   why=[]
   if pixelsha in old_pixels or pixelsha in pixels:why.append('duplicate decoded RGB pixels')
   pixels.add(pixelsha)
   distance=float(prior.query(r['xy'])[0])
   if distance<plan['criteria']['prior_camera_clearance_m']:why.append('actual prior-camera clearance below planned minimum')
   with Image.open(a.study/m['files']['Semantic']['path']) as im:semantic=np.asarray(im.convert('RGB')).astype(np.uint32)
   code=(semantic[:,:,0]<<16)+(semantic[:,:,1]<<8)+semantic[:,:,2];colors=set(np.unique(code).tolist());unknown=colors-palette
   if unknown:why.append('unknown Semantic palette')
   if bluecode in colors:why.append('blue lane unsupported by frozen target policy')
   intensity=m['first_seen']['Intensity']['state'];station=float(intensity['station_m'])
   meta=dict(sample_id=sid,run_id=run.name,role=summary['role'],route_id=summary['route_id'],weather=summary['weather'],hour=summary['hour'],pixel_sha256=pixelsha,prior_distance_m=distance,station_m=station,maximum_file_latency_s=m['maximum_first_seen_latency_s'],files=m['files'],accepted=not why,reasons=why)
   inventory.append(meta)
   if why:rejected.append(meta);continue
   eligible.append((station,r))
  pack='drive_'+summary['role']+'_'+summary['route_id'];all_index.setdefault(pack,dict(simulator_release='26.R1',rows=[]))['rows'].extend(r for _,r in eligible)
  last=-1e9;kept=[]
  for station,row in sorted(eligible):
   if station-last>=a.spacing:kept.append(row);last=station
  thin_index.setdefault(pack,dict(simulator_release='26.R1',rows=[]))['rows'].extend(kept)
  cases.append(dict(run_id=run.name,full=len(eligible),thinned=len(kept),source_snapshot_sha256=source['source_sha256'],completion_source_sha256=summary['source_sha256'],source_binding_matches=source['source_sha256']==summary['source_sha256']))
 expected={(r['route_id'],w,h) for r in routes.values() if r['role'] in a.roles for w,h in r['conditions']}
 missing=sorted(expected-set(selected))
 write(a.output/'index_full.json',all_index);write(a.output/'index_thinned.json',thin_index);write(a.output/'inventory.json',inventory)
 write(a.output/'audit.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(a.plan),prior_index_sha256=sha(a.prior_index),source_sha256=sha(Path(__file__)),roles=a.roles,runs=cases,missing_conditions=missing,full_frames=sum(len(v['rows']) for v in all_index.values()),thinned_frames=sum(len(v['rows']) for v in thin_index.values()),rejected=rejected,minimum_actual_prior_m=min((r['prior_distance_m'] for r in inventory if r['accepted']),default=None),decoded_rgb_unique=len(pixels),raw_channels=['Intensity','Semantic','Depth','Instance'],alignment_scope='Bounded common-save-request moving captures; separate cross-modal alignment audit required. No camera hardware timestamp inferred from file or vehicle timestamp.'))
 print('registered',len(cases),'runs',sum(len(v['rows']) for v in all_index.values()),'full',sum(len(v['rows']) for v in thin_index.values()),'thinned; rejected',len(rejected),'missing',len(missing))
 if missing:raise SystemExit('Incomplete requested roles; registry is a diagnostic partial snapshot')
if __name__=='__main__':main()
