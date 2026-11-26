"""Development-only matched-endpoint comparison and preregistered retention gates."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,collections,hashlib
import numpy as np
from prepare import dump

def pooled(records,head):
 c=np.array([r['counts'][head] for r in records],np.int64).sum(0);return float(c[0]/c.sum()) if c.sum() else None
def metrics(directory):
 rows=[json.loads(s) for s in (directory/'per_frame.jsonl').read_text().splitlines()];dev=[r for r in rows if r['adaptation_role']=='development' and r.get('source_key','').startswith('drive_development_')];retention=[r for r in rows if r['adaptation_role']=='retention']
 perroad={}
 for road in sorted({r['site'] for r in dev}):
  rs=[r for r in dev if r['site']==road];values=[r['counts']['lane'][0]/sum(r['counts']['lane']) for r in rs if sum(r['counts']['lane']) and r['counts']['lane'][0]+r['counts']['lane'][2]>=50]
  perroad[road]=float(np.mean(values))
 return dict(development_macro=float(np.mean(list(perroad.values()))),development_roads=perroad,
             retention={h:pooled(retention,h) for h in ['lane','drivable','white','yellow','stopline']},
             retention_views={v:{h:pooled([r for r in retention if r['view']==v],h) for h in ['lane','drivable']} for v in ['front','left','right']})

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--freeze-provisional',action='store_true');a=p.parse_args();S=a.study
 baseline=metrics(S/'evaluations/f0/development_retention');gates=json.loads((S/'protocol/evaluation_gate_v2_frozen.json').read_text());gate=gates['perception_adoption'];queue=json.loads((S/'protocol/training_queue_frozen.json').read_text())['queue']
 control=S/'protocol/control_queue_v2_frozen.json'
 if control.exists():queue+=json.loads(control.read_text())['queue']
 results=[]
 for item in queue:
  directory=S/'evaluations'/Path(item['run']).name/'development_retention'
  if not (directory/'_SUCCESS').exists():continue
  m=metrics(directory);fail=[];gain=m['development_macro']-baseline['development_macro'];deltas={r:m['development_roads'][r]-baseline['development_roads'][r] for r in m['development_roads']}
  if gain<gate['minimum_macro_road_lane_iou_gain']:fail.append('development_gain')
  if sum(v>0 for v in deltas.values())<gate['minimum_improved_roads']:fail.append('development_road_count')
  if min(deltas.values()) < -gate['maximum_road_lane_iou_loss']:fail.append('development_worst_road')
  for h in ['lane','drivable']:
   if m['retention'][h]<baseline['retention'][h]-gate['maximum_original_pooled_'+h+'_loss']:fail.append('retention_'+h)
  for h in ['white','yellow','stopline']:
   if m['retention'][h] is not None and baseline['retention'][h] is not None and m['retention'][h]<baseline['retention'][h]-gate['maximum_original_class_iou_loss']:fail.append('retention_'+h)
  for view in m['retention_views']:
   for h in ['lane','drivable']:
    if m['retention_views'][view][h]<baseline['retention_views'][view][h]-gate['maximum_original_view_lane_or_drivable_loss']:fail.append('retention_'+view+'_'+h)
  results.append(dict(**item,metrics=m,development_gain=gain,road_deltas=deltas,passes=not fail,failed_gates=fail))
 methods={}
 for arm in sorted({r['arm'] for r in results}):
  rs=[r for r in results if r['arm']==arm];methods[arm]=dict(seeds=len(rs),mean_development_iou=float(np.mean([r['metrics']['development_macro'] for r in rs])),mean_development_gain=float(np.mean([r['development_gain'] for r in rs])),all_seeds_pass=bool(len(rs)==2 and all(r['passes'] for r in rs)))
 report=dict(created_utc=datetime.now(timezone.utc).isoformat(),baseline=baseline,records=results,methods=methods,heldout_accessed=False,gates_sha256=hashlib.sha256((S/'protocol/evaluation_gate_v2_frozen.json').read_bytes()).hexdigest())
 dump(S/'reports/perception_development_comparison.json',report)
 if a.freeze_provisional:
  assert sum(r['arm']!='ReplayOnly' for r in results)==6,'all primary endpoints required'
  if (S/'protocol/provisional_f1_frozen.json').exists():raise ValueError('provisional model already frozen')
  passing=[arm for arm,v in methods.items() if arm!='ReplayOnly' and v['all_seeds_pass']]
  if passing:arm=max(passing,key=lambda arm:methods[arm]['mean_development_iou']);reason='Both seeds pass preregistered retention/development gates'
  else:arm=max([arm for arm in methods if arm!='ReplayOnly'],key=lambda arm:methods[arm]['mean_development_iou']);reason='No repeated passing update; research-only quality-refit candidate, default f0 remains'
  rs=[r for r in results if r['arm']==arm];eligible=[r for r in rs if r['passes']] if passing else rs
  representative=min(eligible,key=lambda r:abs(r['metrics']['development_macro']-methods[arm]['mean_development_iou']))
  checkpoint=Path(representative['run'])/'checkpoint.pt'
  dump(S/'protocol/provisional_f1_frozen.json',dict(selected=representative,method=methods[arm],eligible_for_adoption=bool(passing),reason=reason,
       checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),selected_utc=datetime.now(timezone.utc).isoformat(),heldout_not_used=True,
       interpretation='Tentative new-data endpoint for g1 regeneration; replay-only continuation and heldout checks remain separate.'))
 print(json.dumps(methods,indent=2),flush=True)
if __name__=='__main__':main()
