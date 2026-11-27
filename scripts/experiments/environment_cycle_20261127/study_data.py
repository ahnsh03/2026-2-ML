"""Frozen membership, feature views and buffered geographic folds for this study."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold
from quality_estimation.region_feature_views import feature_matrix

def read(path):return json.loads(Path(path).read_text())
def dump(path,obj):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def lighting(hour):
 h=np.asarray(hour);return np.where((h>=8)&(h<18),0,np.where(((h>=6)&(h<8))|((h>=18)&(h<20)),1,2)).astype(int)
def load_sources(parent):
 z=np.load(Path(parent)/'source_packs_v2.npz');data={k:z[k] for k in z.files};schema=read(Path(parent)/'schema.json');rows=[json.loads(s) for s in (Path(parent)/'protocol/all_sources_v2.jsonl').read_text().splitlines()];assert data['ids'].tolist()==[r['sample_id'] for r in rows]
 return data,schema,rows

def memberships(study,ids):
 index=read(Path(study)/'protocol/quality_index_frozen.json');lookup={s:i for i,s in enumerate(ids)}
 return {key:np.array([lookup[r['sample_id']] for r in part['rows']]) for key,part in index.items()}

def labels(data):
 c=data['counts'];total=c.sum((1,2));q=np.divide(c[:,:,0].sum(1),total,out=np.full(len(total),np.nan),where=total>0);near=c[:,6:8].sum(1);ng=near[:,0]+near[:,2];np_=near[:,0]+near[:,1]
 recall=np.divide(near[:,0],ng,out=np.full(len(total),np.nan),where=ng>0);precision=np.divide(near[:,0],np_,out=np.full(len(total),np.nan),where=np_>0)
 harm=((ng>=50)&(recall<.8))|((np_>=50)&(precision<.8));retained=(ng>=50)&(np_>=50)&(recall>=.8)&(precision>=.8)
 return q,harm,retained

def folds(xy,seed=271127):
 blocks=np.floor(np.asarray(xy)/50).astype(int);groups=np.array([str(tuple(x)) for x in blocks]);out=[]
 for train,val in GroupKFold(3).split(xy,groups=groups):
  distance=cKDTree(np.asarray(xy)[val]).query(np.asarray(xy)[train])[0];train=train[distance>=25]
  if len(train)<100:raise ValueError('insufficient buffered training rows')
  out.append((train,val))
 return out

def quality_views(data,schema,base):
 fp=['near_left.lane_drivable_disagreement','near_right.lane_drivable_disagreement','global.lane_drivable_disagreement','near.soft_outside','near_left.internal_gap','near_right.internal_gap']
 photo=['intensity','intensity_std','dark_fraction','glare_fraction','intensity_p90_minus_p10','saturation_median','blue_minus_red','hue_cos','hue_sin']
 env=[n for n in schema['rgb'] if n.startswith(('upper_top_','hood.')) and n.split('.')[1] in photo]
 unique=lambda x:list(dict.fromkeys(x))
 cols=dict(base28=base,fp34=unique(base+fp),env52=unique(base+env),morph49=unique(base+schema['morphology']))
 return {k:(v,feature_matrix(data,schema,v)) for k,v in cols.items()}

def macro(values,groups):
 values=np.asarray(values);groups=np.asarray(groups);valid=np.isfinite(values)
 scores=[float(np.mean(values[(groups==g)&valid])) for g in np.unique(groups) if ((groups==g)&valid).any()]
 if not scores:raise ValueError('macro metric has no defined groups')
 return float(np.mean(scores))
