"""Dependency-free file/context verification of the current RGB review handoff."""
from pathlib import Path
import argparse,json
from verify_inference_release import digest


def verify(root):
 root=Path(root).resolve();spec=json.loads((root/'release_manifest.json').read_text())
 if spec['version']!='environment-inference-release-1':raise ValueError('unknown environment release')
 if not {'models/review/bundle_manifest.json','examples/manifest.jsonl'}.issubset(spec['files']):raise ValueError('release inventory incomplete')
 for name,expected in spec['files'].items():
  path=(root/name).resolve()
  try:path.relative_to(root)
  except ValueError as exc:raise ValueError('release file escapes root') from exc
  if digest(path)!=expected:raise ValueError('release file changed: '+name)
 identity=spec['perception_identity'];bundle=json.loads((root/'models/review/bundle_manifest.json').read_text())
 if identity['kind'] not in ['baseline','adaptation']:raise ValueError('unknown perception identity')
 if bundle['perception_identity']!=identity:raise ValueError('quality/perception identity differs')
 for name,expected in bundle['artifact_sha256'].items():
  if spec['files'].get('models/review/'+name)!=expected:raise ValueError('review artifact inventory differs')
 if identity['kind']=='adaptation':
  run=root/'models/perception/run.json';checkpoint=run.with_name('checkpoint.pt')
  if digest(run)!=identity['adaptation_run_sha256'] or run.with_name('_SUCCESS').read_text().strip()!=digest(run):raise ValueError('adaptation lineage differs')
  for name,expected in json.loads(run.read_text())['artifact_sha256'].items():
   if spec['files'].get('models/perception/'+name)!=expected:raise ValueError('adaptation artifact inventory differs')
 else:checkpoint=root/'models/perception/best_mean.pt'
 if digest(checkpoint)!=identity['checkpoint_sha256']:raise ValueError('perception checkpoint differs')
 rows=[json.loads(line) for line in (root/'examples/manifest.jsonl').read_text().splitlines()]
 if len(rows)!=spec['rgb_examples'] or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('example membership differs')
 for row in rows:
  if set(row)!={'sample_id','image','view','expected_sha256'} or set(row['expected_sha256'])!={'rgb'} or row['view']!='front':raise ValueError('front RGB-only examples required')
  image=(root/'examples/rgb'/row['image']).resolve();image.relative_to(root/'examples/rgb')
  if digest(image)!=row['expected_sha256']['rgb']:raise ValueError('RGB example changed')
 return dict(status='verified',adoption_status=spec.get('adoption_status','not_recorded'),files=len(spec['files']),rgb_examples=len(rows),source_git=spec['source_git'],perception_identity=identity,contains_raw_ground_truth=False,contains_training_lineage=identity['kind']=='adaptation',scope='File/context verification; actual standalone inference and offline adoption are reported separately.')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);print(json.dumps(verify(p.parse_args().root),indent=2))
