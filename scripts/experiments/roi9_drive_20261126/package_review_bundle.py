"""Copy frozen review artifacts and RGB-only smoke inputs into a portable bundle."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,shutil
from feature_study import read,save,sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['study','old','repo','samples','data-root','output']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();n=a.study;out=a.output;out.mkdir(parents=True,exist_ok=False);selection=read(n/'feature_study/selection_frozen.json')
 files={'feature_study/selection_frozen.json':n/'feature_study/selection_frozen.json','policy/policy_frozen.json':n/'review_policy_v1/policy_frozen.json','training_membership.json':a.old/'frozen_bundle_final/training_membership.json'}
 for record in [selection['chosen'],selection.get('far_auxiliary')]:
  if record:files['feature_study/'+record['model']]=n/'feature_study'/record['model']
 for name in ['calibration_frozen.json','environment_profile.json','global_moments_reference.json']:files['calibration/'+name]=n/'calibration_v1'/name
 for relative,source in files.items():
  target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
 rows=[json.loads(line) for line in a.samples.read_text().splitlines() if line.strip()];smoke=[]
 for i,row in enumerate(rows):
  image=a.data_root/row['image'];assert sha(image)==row['expected_sha256']['rgb'];relative=f'smoke/{i:03d}.png';target=out/relative;target.parent.mkdir(exist_ok=True);shutil.copyfile(image,target)
  smoke.append(dict(row,image=relative))
 (out/'rgb_manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in smoke))
 source_names=['drive_review.py','drive_calibration.py','spatial_gp.py','spatial_risk.py','spatial_composition.py','spatial_features.py','selected_region_features.py','region_candidates.py','region_feature_views.py','road_structure_features.py','roi10.py','infer_review_images.py','review_calibrated.py']
 c=read(out/'calibration/calibration_frozen.json')
 save(out/'bundle_manifest.json',dict(version='drive-review-bundle-1',created_utc=datetime.now(timezone.utc).isoformat(),artifact_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},source_sha256={'src/quality_estimation/'+name:sha(a.repo/'src/quality_estimation'/name) for name in source_names},profile=c['profile'],regions_sha256=c['regions_sha256'],checkpoint_sha256=c['checkpoint_sha256'],simulator_release=c['simulator_release'],primary_dimensions=len(c['columns']),quality_union_dimensions=len(selection['required_feature_union']),RGB_smoke_rows=len(rows),scope='Frozen experimental spatial-review candidate; consult prospective evaluation before adoption. Perception checkpoint/software are supplied separately; raw GT and simulator vendor files are not bundled. Membership provenance is private data, not an inference input.'))
 print('packaged frozen spatial review',len(rows),'RGB smoke rows',flush=True)
if __name__=='__main__':main()
