"""Export aggregate evidence only; keep raw rows, paths, weights and logs private."""
from pathlib import Path
import argparse,json,shutil
from feature_study import read,save,sha

def aggregate(value):
 if isinstance(value,dict):return {key:aggregate(item) for key,item in value.items() if key not in ['selected_ids','development_ids','raw_files_sha256','source_packs_sha256']}
 if isinstance(value,list):return [aggregate(item) for item in value]
 return value
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();n=a.study;a.output.mkdir(parents=True,exist_ok=False)
 assert (n/'VERIFICATION_COMPLETE').exists() and (n/'marked_support/FIGURES_COMPLETE').exists()
 selection=read(n/'feature_study/selection_frozen.json');cal=read(n/'calibration_v1/calibration_frozen.json');policy=read(n/'review_policy_v1/policy_frozen.json');audit=read(n/'dataset_final_audit/summary.json')
 result=dict(perception_checkpoint_sha256=cal['checkpoint_sha256'],regions_sha256=cal['regions_sha256'],feature_selection=dict(primary_columns=selection['chosen']['columns'],far_columns=selection['far_auxiliary']['columns'] if selection['far_auxiliary'] else [],quality_union=selection['required_feature_union'],primary_model_sha256=selection['chosen']['model_sha256'],far_model_sha256=selection['far_auxiliary']['model_sha256'] if selection['far_auxiliary'] else None,development_rows=selection['development_rows'],source_sha256=selection['source_sha256'],results=[{key:r[key] for key in ['view','kind','dimensions','development_macro_mae','development_by_road']} for r in selection['results']],auxiliary_diagnostic=selection['auxiliary_diagnostic']),primary_test=aggregate(read(n/'prospective_test_v1/results.json')),supported_followup=aggregate(read(n/'marked_support/evaluation/results.json')),condition_shift=aggregate(read(n/'challenge/evaluation/results.json')),calibration_development=dict(selected=aggregate(cal['selected_cv']),moments=cal['moments'],joint_map=cal['joint_map'],environment_thresholds=cal['environment_thresholds'],event_definition=cal['event_definition']),policy=aggregate(policy),policy_search=aggregate(read(n/'review_policy_v1/search.json')),main_data_audit={key:value for key,value in audit.items() if key!='run_condition_records'},actual_quality_training_clearance=read(n/'actual_quality_training_clearance.json'),replay_GT_verification=read(n/'replay_GT_verification/summary.json'),runtime_benchmark=read(n/'benchmark_review_cpu.json'),training_reconstruction=read(n/'selected_GP_membership_reconstruction.json'),source_stamp_sensitivity=read(n/'source_stamp_sensitivity.json'),master_inventory=read(n/'global_registry/audit.json'),scope='Primary and post-primary GT-supported cohorts remain separate. Empirical diagnostic probabilities, no live ROS,40GPU or new perception-finetuning claim.')
 save(a.output/'results.json',result)
 for root,prefix in [(n/'figures_final','main_'),(n/'marked_support/figures','supported_')]:
  for file in root.iterdir():
   if file.suffix in ['.png','.svg']:shutil.copyfile(file,a.output/(prefix+file.name))
 for file in (n/'marked_support/case_figures').glob('*_day_night.png'):shutil.copyfile(file,a.output/file.name)
 save(a.output/'artifact_sha256.json',{p.name:sha(p) for p in a.output.iterdir() if p.is_file()})
 print('public aggregate export',a.output,flush=True)
if __name__=='__main__':main()
