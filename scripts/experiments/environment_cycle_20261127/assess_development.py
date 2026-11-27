"""Assess frozen Dev/retention gates, schedule paired repeats, freeze f1."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,collections
import numpy as np
from study_data import read,dump,sha

def summary(rows):
 def pooled(head):
  c=np.array([r['counts'][head] for r in rows]).sum(0);return float(c[0]/c.sum()) if c.sum() else None
 return {head:pooled(head) for head in ['lane','drivable','white','yellow','stopline']}
def measures(path):
 rows=[json.loads(x) for x in (path/'per_frame.jsonl').read_text().splitlines()];dev=[r for r in rows if (r.get('source_key') or '').startswith('drive_development_')];old=[r for r in rows if r['adaptation_role']=='retention'];road={}
 for site in sorted({r['site'] for r in dev}):
  c=np.array([r['counts']['lane'] for r in dev if r['site']==site]);valid=(c[:,0]+c[:,2]>=50)&(c.sum(1)>0);road[site]=float(np.mean(c[valid,0]/c[valid].sum(1)))
 return dict(primary_macro_road_iou=float(np.mean(list(road.values()))),by_road=road,retention=summary(old),views={view:summary([r for r in old if r['view']==view]) for view in ['front','left','right']})
def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);S=p.parse_args().study;plan=read(S/'protocol/training_plan.json');f0=S/'evaluations/f0/development_retention'
 if not (f0/'per_frame.jsonl').exists():return
 baseline=measures(f0);rules=read(S/'protocol/study_design_frozen.json')['nn_adoption'];records=[]
 for item in plan['queue']:
  for alpha in [1.,.25,.5,.75]:
   name=item['name'] if alpha==1. else item['name']+f'_a{int(alpha*100):03d}';evalpath=S/'evaluations'/name/'development_retention'
   if not (evalpath/'_SUCCESS').exists():continue
   m=measures(evalpath);gain=m['primary_macro_road_iou']-baseline['primary_macro_road_iou'];deltas={g:m['by_road'][g]-baseline['by_road'][g] for g in m['by_road']};fail=[]
   if gain<rules['new_primary_macro_iou_gain_min']:fail.append('new_gain')
   if sum(x>0 for x in deltas.values())<rules['positive_roads_min']:fail.append('road_count')
   if min(deltas.values()) < -rules['new_road_iou_loss_max']:fail.append('worst_road')
   for head in ['lane','drivable','white','yellow','stopline']:
    limit=rules['old_'+head+'_pooled_loss_max'] if head in ['lane','drivable'] else rules['old_class_view_iou_loss_max']
    if baseline['retention'][head] is not None and m['retention'][head] is not None and m['retention'][head] < baseline['retention'][head]-limit:fail.append('old_'+head)
   for view in m['views']:
    for head in ['lane','drivable']:
     if m['views'][view][head]<baseline['views'][view][head]-rules['old_class_view_iou_loss_max']:fail.append(view+'_'+head)
   records.append(dict(name=name,arm=item['arm'],seed=item['seed'],alpha=alpha,run=str(Path(item['run']) if alpha==1. else S/'derived'/name),metrics=m,gain=gain,deltas=deltas,passes=not fail,failed_gates=fail))
 dump(S/'reports/neural_development_partial.json',dict(baseline=baseline,records=records,confirmation_opened=False))
 first=[r for r in records if r['seed']==271127]
 if len(first)==16 and not (S/'protocol/repeat_choice_frozen.json').exists():
  eligible=[r for r in first if r['arm']!='RandomFull' and r['passes']];winner=max(eligible or [r for r in first if r['arm']!='RandomFull'],key=lambda r:(r['gain'],r['alpha']))
  dump(S/'protocol/repeat_choice_frozen.json',dict(selected_arm=winner['arm'],candidate=winner,eligible=bool(eligible),created_utc=datetime.now(timezone.utc).isoformat(),rule='Repeat Random/RF/GP; one balanced diagnostic and one full-original continuation control complete eight runs. No fresh confirmation opened.'))
  for arm in ['RandomFull','RFGlobalFull','GPGlobalFull']:
   name=arm+'_s271128';plan['queue'].append(dict(name=name,arm=arm,seed=271128,manifest=str(S/f'protocol/{arm}_cached.jsonl'),run=str(S/'runs'/name)))
  plan['queue_closed']=True;dump(S/'protocol/training_plan.json',plan);print('REPEAT FROZEN',winner['arm'],flush=True)
 if plan.get('queue_closed') and len(records)==len(plan['queue'])*4 and not (S/'protocol/provisional_f1_frozen.json').exists():
  groups=collections.defaultdict(list)
  for r in records:groups[(r['arm'],r['alpha'])].append(r)
  eligible=[v for v in groups.values() if len(v)==2 and all(r['passes'] for r in v)]
  choice=max(eligible or list(groups.values()),key=lambda v:(np.mean([r['gain'] for r in v]),len(v),v[0]['alpha']))
  representative=min(choice,key=lambda r:abs(r['gain']-np.mean([x['gain'] for x in choice])))
  dump(S/'protocol/provisional_f1_frozen.json',dict(selected=representative,records=choice,eligible_for_adoption=bool(eligible),checkpoint_sha256=sha(Path(representative['run'])/'checkpoint.pt'),selected_utc=datetime.now(timezone.utc).isoformat(),confirmation_not_used=True))
  print('F1 FROZEN',representative['name'],'eligible',bool(eligible),flush=True)
if __name__=='__main__':main()
