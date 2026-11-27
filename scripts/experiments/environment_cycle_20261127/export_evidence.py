"""Publish aggregates and figures only; keep raw IDs/locations/weights external."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,shutil,re
from study_data import read,dump,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();S=a.study;out=a.output;out.mkdir(parents=True,exist_ok=True)
 if not (S/'reports/final_evidence_complete.json').exists():raise ValueError('do not export an unfinished offline result as final')
 files=['f0_quality/review_tie_study.json','g1_quality/review_tie_study.json','f0_quality/uncertainty_diagnostics.json','f0_quality/interval_calibration_study.json','g1_quality/uncertainty_diagnostics.json','g1_quality/interval_calibration_study.json','figures/lighting_support.json','reports/selection_bank_overlap.json','verification/standalone_f0/runtime_verification.json','verification/standalone_g1/runtime_verification.json','reports/learned_representation_cluster_controls.json','reports/original_sampler_exposure_audit.json','reports/replay_hash_io_verification.json','reports/code_validation_current.json','reports/original_capture_metadata_audit.json','reports/clustering_implementation_reproduction.json','reports/original_replay_audit.json','reports/f0_training_provenance.json','reports/full_original_cache_audit.json','reports/admission_summary.json','reports/admission_integrity_before_training.json','reports/environment_context_audit.json','reports/role_transition.json','reports/selection_lineage_no_candidate_gt.json','reports/neural_development_partial.json','reports/full_replay_control_complete.json','reports/fresh_confirmation_assessment.json','reports/relay_recovery.json','reports/code_validation_initial.json','protocol/study_design_frozen.json','reports/identical_policy_banks.json','deployment/candidate_pair.json','f0_quality/quality_selection_frozen.json','g1_quality/quality_selection_frozen.json','g1_quality/review_policy_frozen.json','environment/environment_selection_frozen.json','confirmation/registry/audit.json','verification/g1_quality/runtime_verification.json']
 def scrub(value):
  if isinstance(value,dict):return {k:scrub(v) for k,v in value.items()}
  if isinstance(value,list):return [scrub(v) for v in value]
  if isinstance(value,str):
   value=value.replace(str(S),'${STUDY_ROOT}').replace(str(a.parent),'${PARENT_STUDY}')
   if re.search(r'(?:^|\s)/(?:home|mnt)/',value) or re.search(r'\b[A-Za-z]:[\\/]',value):raise ValueError('unmapped absolute asset path in public aggregate')
  return value
 inventory={}
 for name in files:
  path=S/name
  if not path.exists():continue
  dest=out/name;dump(dest,scrub(read(path)));inventory[str(dest.relative_to(out))]=dict(sha256=sha(dest),source_sha256=sha(path))
 for path in (S/'figures').glob('*'):
  if path.suffix not in ['.png','.svg']:continue
  dest=out/'figures'/path.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest);inventory[str(dest.relative_to(out))]=dict(sha256=sha(dest),source_sha256=sha(path))
 dump(out/'evidence_inventory.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),artifacts=inventory,raw_data_copied=False,weights_published=False,scope='Aggregate paired simulator research evidence; source manifests and raw locations remain local.'))
 print('PUBLIC AGGREGATES',len(inventory),flush=True)
if __name__=='__main__':main()
