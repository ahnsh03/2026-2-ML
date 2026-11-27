"""Exercise the Dev assessor's real file/metric/queue path without GPU data."""
from pathlib import Path
import json,subprocess,sys,tempfile,unittest


class RepeatAllocationTest(unittest.TestCase):
 def test_competitive_balanced_bank_can_repeat_within_eight_run_budget(self):
  repo=Path(__file__).resolve().parents[1]
  with tempfile.TemporaryDirectory() as tmp:
   study=Path(tmp);(study/'protocol').mkdir()
   def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
   rules=dict(new_primary_macro_iou_gain_min=.005,positive_roads_min=2,new_road_iou_loss_max=.02,old_lane_pooled_loss_max=.005,old_drivable_pooled_loss_max=.005,old_class_view_iou_loss_max=.01)
   write(study/'protocol/study_design_frozen.json',{'nn_adoption':rules})
   arms={'RandomFull':70,'RFGlobalFull':71,'GPGlobalFull':72,'TagBalancedFull':75};queue=[dict(name=arm+'_s271127',arm=arm,seed=271127,run=str(study/'runs'/(arm+'_s271127'))) for arm in arms]
   write(study/'protocol/training_plan.json',dict(queue=queue,queue_closed=False))
   def evaluation(name,tp):
    records=[]
    for i in range(3):
     records.append(dict(source_key='drive_development_'+str(i),site=str(i),view='front',adaptation_role='development',counts={h:[tp,0,100-tp] for h in ['lane','drivable','white','yellow','stopline']}))
     records.append(dict(source_key='retention',site='old',view=['front','left','right'][i],adaptation_role='retention',counts={h:[80,0,20] for h in ['lane','drivable','white','yellow','stopline']}))
    out=study/'evaluations'/name/'development_retention';out.mkdir(parents=True);(out/'per_frame.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records));(out/'_SUCCESS').write_text('complete\n')
   evaluation('f0',60)
   for item in queue:
    for alpha in [1.,.25,.5,.75]:evaluation(item['name'] if alpha==1 else item['name']+f'_a{int(alpha*100):03d}',arms[item['arm']])
   subprocess.run([sys.executable,str(repo/'scripts/experiments/environment_cycle_20261127/assess_development.py'),'--study',str(study)],check=True,capture_output=True,text=True)
   choice=json.loads((study/'protocol/repeat_choice_frozen.json').read_text());plan=json.loads((study/'protocol/training_plan.json').read_text())
   self.assertEqual(choice['repeated_arms'],['RandomFull','TagBalancedFull','GPGlobalFull']);self.assertTrue(plan['queue_closed']);self.assertEqual(len(plan['queue']),7);self.assertEqual(len(plan['queue'])+1,8)
   self.assertEqual(sum(item['arm']=='RFGlobalFull' for item in plan['queue']),1)
if __name__=='__main__':unittest.main()
