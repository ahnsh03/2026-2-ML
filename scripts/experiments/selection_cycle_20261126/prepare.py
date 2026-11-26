"""Freeze adaptation admissions, original replay and geographically buffered diagnostics."""
from pathlib import Path
import argparse,json,collections,hashlib
import numpy as np
from scipy.spatial import cKDTree
from quality_estimation.region_feature_views import feature_matrix
from quality_estimation.spatial_gp import SpatialQualityGP

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def jsonl(path,rows):path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
def source_role(key,quality_member=False):
 if quality_member:return 'quality_fit'
 if key.startswith('drive_development_') or key in ['z05','z10']:return 'development'
 # The shift prefix must be resolved before the broader calibration prefix.
 if key.startswith(('drive_test_','drive_confirmation_','drive_calibration_shift_')) or key in ['z04','z07','z13']:return 'heldout_diagnostic'
 if key.startswith(('drive_calibration_','drive_support_test_')) or key in ['z03','z12']:return 'calibration'
 return None
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--study',type=Path,required=True);p.add_argument('--original-data',type=Path,required=True);a=p.parse_args();E=a.root;out=a.study;out.mkdir(exist_ok=True);(out/'protocol').mkdir(exist_ok=True)
 idx=json.loads((E/'ml_roi9_drive_20261126/global_registry_v2/index_full.json').read_text());art=json.loads((E/'ml_roi9_drive_20261126/global_registry_v2/pack_artifacts.json').read_text())
 parts={};rows=[];schema=None
 for key,pack in idx.items():
  with np.load(E/art[key]['artifact']) as z:
   if z['ids'].tolist()!=[r['sample_id'] for r in pack['rows']]:raise ValueError('pack/source row order mismatch')
   for name in z.files:parts.setdefault(name,[]).append(z[name])
  rows.extend([dict(r,source_key=key) for r in pack['rows']])
  if schema is None:schema=json.loads((E/art[key]['schema']).read_text())
 data={k:np.concatenate(v) for k,v in parts.items()};assert len(set(data['ids']))==len(rows)
 heldkeys=[k for k in idx if k.startswith(('drive_','release26_','geo_block_')) or k in ['z03','z04','z05','z07','z10','z12','z13']]
 held=np.array([r['xy'] for k in heldkeys for r in idx[k]['rows']]);distance=cKDTree(held).query(data['xy'])[0]
 candidatekeys=[k for k in idx if k.startswith(('nov23_m','archive_')) or k in ['stream_new','mixed_a','mixed_b']]
 pool=np.array([r['source_key'] in candidatekeys for r in rows])&(distance>=25)
 # Never use a quality-fit image as an adaptation admission; it will be relabelled for g1.
 quality_ids={r['sample_id'] for r in json.loads((E/'ml_roi9_gp_20261125/frozen_bundle_final/training_membership.json').read_text())['rows']}
 pool &= ~np.isin(data['ids'],list(quality_ids))
 role=np.full(len(rows),'historical_not_admitted',dtype='<U40');role[pool]='candidate_pool'
 for i,r in enumerate(rows):
  resolved=source_role(r['source_key'],r['sample_id'] in quality_ids)
  if resolved:role[i]=resolved
 for i,r in enumerate(rows):r.update(adaptation_role=str(role[i]),image='e/'+r['image'],semantic='e/'+r['semantic'],parent_session_id=r.get('group',r['source_key']),source_role='after_f0_training')
 assert np.all(distance[pool]>=25)
 gp=SpatialQualityGP.load(E/'ml_roi9_gp_20261125/frozen_bundle_final/quality_spatial_gp.npz');x=feature_matrix(data,schema,gp.feature_names_.tolist());means=[];cov=[]
 for start in range(0,len(x),256):
  m,c=gp.predict(x[start:start+256]);means.append(m);cov.append(c)
  if start%2048==0:print('g0 predicted',start,'/',len(x),flush=True)
 data.update(g0_mean=np.concatenate(means),g0_covariance=np.concatenate(cov),role=role)
 np.savez_compressed(out/'source_packs.npz',**data);dump(out/'schema.json',schema);jsonl(out/'protocol/all_sources.jsonl',rows);jsonl(out/'protocol/candidate_pool.jsonl',[r for r,b in zip(rows,pool) if b])
 # Preserve the original validation cohort and draw replay from the frozen f0 train manifest.
 D=a.original_data;prefix=str(D.resolve().relative_to(Path('/mnt')));version=D/'dataset_versions/twinlite_morai_v6_road_marking_actor_negative'
 def native(r,split):
  rr=dict(sample_id=r['sample_id'],run_id=r['run_id'],parent_session_id=r['run_id'],frame_id=r['frame_id'],view=r['view'],
          image=prefix+'/'+r['image'],semantic=prefix+'/datasets/'+r['run_id']+'/frames/semantic/'+r['view']+'/'+Path(r['image']).name,
          source_role='f0_train' if split=='train' else 'historical_dev',adaptation_role='replay' if split=='train' else 'retention',original_hw=[720,1280])
  if not (Path('/mnt')/rr['semantic']).exists():raise ValueError('missing original Semantic: '+rr['sample_id'])
  return rr
 train=[json.loads(s) for s in (version/'train.jsonl').read_text().splitlines()];val=[json.loads(s) for s in (version/'val.jsonl').read_text().splitlines()]
 rng=np.random.default_rng(261126);groups=collections.defaultdict(list)
 for r in train:groups[(r['run_id'],r['view'])].append(r)
 picked=[];quotas={};keys=sorted(groups)
 for j,key in enumerate(keys):
  n=4096//len(keys)+(j<4096%len(keys));g=groups[key];chosen=np.linspace(0,len(g)-1,n,dtype=int);picked.extend(native(g[i],'train') for i in chosen);quotas[str(key)]=n
 assert len(picked)==4096 and len(set(r['sample_id'] for r in picked))==4096
 jsonl(out/'protocol/replay_sources.jsonl',picked);jsonl(out/'protocol/retention_sources.jsonl',[native(r,'val') for r in val])
 dump(out/'protocol/roles_frozen.json',dict(roles=dict(collections.Counter(role)),candidate_pool=int(pool.sum()),minimum_adaptation_to_reserved_distance_m=float(distance[pool].min()),
      replay=len(picked),retention=len(val),replay_quotas=quotas,original_f0_train_count=len(train),original_f0_front_count=sum(r['view']=='front' for r in train),
      evaluation_scope='Previously opened cohorts reserved from all new adaptation; paired f1 diagnostic, not a pristine unseen final test.',
      source_index_sha256=hashlib.sha256((E/'ml_roi9_drive_20261126/global_registry_v2/index_full.json').read_bytes()).hexdigest()))
 print('FROZEN',collections.Counter(role),flush=True)
if __name__=='__main__':main()
