"""Replace new-data slots with distinct known f0 train images, retaining common replay slots."""
from pathlib import Path
import argparse,json,collections
import numpy as np
from prepare import dump,jsonl

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--original-data',type=Path,required=True);a=p.parse_args();S=a.study;D=a.original_data;prefix=str(D.resolve().relative_to('/mnt'))
 common=[json.loads(s) for s in (S/'protocol/replay_sources.jsonl').read_text().splitlines()];excluded={r['sample_id'] for r in common};groups=collections.defaultdict(list)
 for row in map(json.loads,(D/'dataset_versions/twinlite_morai_v6_road_marking_actor_negative/train.jsonl').read_text().splitlines()):
  if row['sample_id'] not in excluded and row['view']=='front':groups[(row['run_id'],row['view'])].append(row)
 picked=[];keys=sorted(groups)
 quotas={key:0 for key in keys}
 while sum(quotas.values())<3072:
  for key in keys:
   if sum(quotas.values())>=3072:break
   if quotas[key]<len(groups[key]):quotas[key]+=1
 for j,key in enumerate(keys):
  g=groups[key];n=quotas[key]
  for i in np.linspace(0,len(g)-1,n,dtype=int):
   r=g[i];picked.append(dict(sample_id=r['sample_id'],run_id=r['run_id'],parent_session_id=r['run_id'],frame_id=r['frame_id'],view=r['view'],
        image=prefix+'/'+r['image'],semantic=prefix+'/datasets/'+r['run_id']+'/frames/semantic/'+r['view']+'/'+Path(r['image']).name,
        source_role='f0_train',adaptation_role='replay_replacement',original_hw=[720,1280]))
 assert len(picked)==3072 and len({r['sample_id'] for r in picked})==3072 and not excluded.intersection(r['sample_id'] for r in picked)
 jsonl(S/'protocol/replay_control_sources_v2.jsonl',picked)
 dump(S/'protocol/replay_control_v2_frozen.json',dict(purpose='Same 8 common replay slots; replace 8 selected front-camera new-data slots with known original front training data.',
      replacement_rows=len(picked),common_replay_rows=len(common),seeds=[261126,261127],steps=24000,initial_checkpoint='f0',new_images=0,
      teacher_constraint='Same frozen-f0 distillation on common replay slots only, as every new-data arm.',endpoint='fixed; no final-test checkpoint selection',camera_view='front',quotas={str(k):v for k,v in quotas.items()},
      amendment='v1 replacement included side cameras; retire it before any control training. v2 matches camera proportions of new-data arms.',
      scientific_scope='Distinguish selected novel labels from an equal-step continuation on original data. Frozen before any f1 development evaluation.'))
if __name__=='__main__':main()
