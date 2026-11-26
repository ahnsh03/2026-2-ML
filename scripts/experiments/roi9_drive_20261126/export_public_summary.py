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
 if (n/'confirmation/runs').exists():assert all((n/name).exists() for name in ['FINAL_CONFIRMATION_VERIFIED','CONFIRMATION_ANALYSIS_COMPLETE','FINAL_RESTORATION_COMPLETE'])
 selection=read(n/'feature_study/selection_frozen.json');cal=read(n/'calibration_v1/calibration_frozen.json');policy=read(n/'review_policy_v1/policy_frozen.json');audit=read(n/'dataset_audit_final/summary.json')
 result=dict(perception_checkpoint_sha256=cal['checkpoint_sha256'],regions_sha256=cal['regions_sha256'],feature_selection=dict(primary_columns=selection['chosen']['columns'],far_columns=selection['far_auxiliary']['columns'] if selection['far_auxiliary'] else [],quality_union=selection['required_feature_union'],primary_model_sha256=selection['chosen']['model_sha256'],far_model_sha256=selection['far_auxiliary']['model_sha256'] if selection['far_auxiliary'] else None,development_rows=selection['development_rows'],source_sha256=selection['source_sha256'],results=[{key:r[key] for key in ['view','kind','dimensions','development_macro_mae','development_by_road']} for r in selection['results']],auxiliary_diagnostic=selection['auxiliary_diagnostic']),primary_test=aggregate(read(n/'prospective_test_v1/results.json')),supported_followup=aggregate(read(n/'marked_support/evaluation/results.json')),condition_shift=aggregate(read(n/'challenge/evaluation/results.json')),calibration_development=dict(selected=aggregate(cal['selected_cv']),moments=cal['moments'],joint_map=cal['joint_map'],environment_thresholds=cal['environment_thresholds'],event_definition=cal['event_definition']),policy=aggregate(policy),policy_search=aggregate(read(n/'review_policy_v1/search.json')),main_data_audit={key:value for key,value in audit.items() if key!='run_condition_records'},actual_quality_training_clearance=read(n/'actual_quality_training_clearance.json'),replay_GT_verification=read(n/'replay_GT_verification/summary.json'),runtime_benchmark=read(n/'benchmark_review_cpu.json'),training_reconstruction=read(n/'selected_GP_membership_reconstruction.json'),source_stamp_sensitivity=read(n/'source_stamp_sensitivity.json'),master_inventory=read(n/'global_registry/audit.json'),scope='Primary and post-primary GT-supported cohorts remain separate. Empirical diagnostic probabilities, no live ROS,40GPU or new perception-finetuning claim.')
 profile=read(n/'calibration_v1/environment_profile.json');search=read(n/'review_policy_v1/search.json')
 result.update(calibration_development=dict(selected_cv=aggregate(cal['selected_cv']),moments=cal['moments'],joint_map=cal['joint_map'],global_map=cal['global_map'],environment_thresholds=cal['environment_thresholds'],event_definition=cal['event_definition']),policy=aggregate(policy),policy_search={key:search[key] for key in ['BO_best','random_best','adopted_for_prospective_test']},environment=dict(ordered_features=profile['ordered_features'],dimensions=len(profile['ordered_features']),nonconstant_dimensions=len(profile['selected']),fine_centers=len(profile['centers']),coarse_policy_groups=len(set(profile['policy_map'])),scope='Frozen previous official e-means centers; coarse groups merge centers using RGB geometry only. No semantic weather labels used online.'))
 for key,path in [('secondary_expected_deficit','expected_deficit_secondary/supported_results.json'),('secondary_uncertainty_ablation','expected_deficit_secondary/ablation_results.json'),('probability_saturation','probability_saturation_primary.json'),('fresh_confirmation','confirmation/evaluation/results.json'),('confirmation_adoption','confirmation/adoption.json'),('confirmation_clearance','confirmation/clearance.json'),('confirmation_context','confirmation/context_results.json'),('supported_context','marked_support/context_results.json'),('confirmation_sensitivity','confirmation/sensitivity/results.json'),('current_data_roles','role_ledger_v2/summary.json'),('fresh_RGB_replay','confirmation/replay_GT_verification_v2/summary.json'),('master_inventory_after_confirmation','global_registry_v2/audit.json')]:
  if (n/path).exists():result[key]=aggregate(read(n/path))
 if (n/'calibration_v2_final/calibration_frozen.json').exists():
  c2=read(n/'calibration_v2_final/calibration_frozen.json')
  result['spatial_calibration_v2']={key:aggregate(c2[key]) for key in ['calibration_rows','calibration_roads','moments','selected_cv','Q_moments_unchanged','near_far_scale_floor','retired_supported_followup']}
 if (n/'watchdog_test_v4/result.json').exists():result['watchdog_fault_probe']=read(n/'watchdog_test_v4/result.json')
 if (n/'dataset_audit_with_confirmation/summary.json').exists():result['all_drive_data_audit']={k:v for k,v in read(n/'dataset_audit_with_confirmation/summary.json').items() if k!='run_condition_records'}
 save(a.output/'results.json',result)
 for root,prefix in [(n/'figures_final','main_'),(n/'marked_support/figures','supported_'),(n/'confirmation/figures','confirmation_'),(n/'motion_figures_final_v2','motion_'),(n/'confirmation/motion_figures','drive_')]:
  if not root.exists():continue
  for file in root.iterdir():
   if file.suffix in ['.png','.svg']:shutil.copyfile(file,a.output/(prefix+file.name))
 for file in (n/'marked_support/case_figures_corrected').glob('*_day_night.png'):shutil.copyfile(file,a.output/file.name)
 for file in (n/'confirmation/case_figures').glob('*_day_night.png'):shutil.copyfile(file,a.output/file.name)
 for file in (n/'confirmation/probability_cases').glob('class_*.png'):shutil.copyfile(file,a.output/('probability_'+file.name))
 if (n/'roi9_drive_overlay/roi9_native.png').exists():shutil.copyfile(n/'roi9_drive_overlay/roi9_native.png',a.output/'roi9_native.png')
 if (n/'figures_cases_corrected/drive26_r02_day_night.png').exists():shutil.copyfile(n/'figures_cases_corrected/drive26_r02_day_night.png',a.output/'main_sparse_markings.png')
 save(a.output/'artifact_sha256.json',{p.name:sha(p) for p in a.output.iterdir() if p.is_file()})
 print('public aggregate export',a.output,flush=True)
if __name__=='__main__':main()
