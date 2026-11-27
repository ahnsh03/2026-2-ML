"""Package the current Q/environment model and exact perception lineage for a team."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,shutil,subprocess,tarfile
from package_inference_release import git
from verify_inference_release import digest
from verify_environment_release import verify
from quality_estimation.environment_review import EnvironmentReview
from quality_estimation.perception_identity import resolve_quality_perception


def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--rgb-root',type=Path,required=True);p.add_argument('--rgb-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--examples',type=int,default=8);p.add_argument('--adoption-status',required=True)
 source=p.add_mutually_exclusive_group(required=True);source.add_argument('--checkpoint',type=Path);source.add_argument('--adaptation-run',type=Path);a=p.parse_args();repo=a.repo.resolve();out=a.output.resolve()
 source_paths=['src','config','docker','scripts','.gitmodules','ros/src/data_collection/src/data_collection']
 if git(repo,'status','--porcelain','--',*source_paths):raise ValueError('commit the packaged source before release; documentation changes may remain')
 checkpoint,identity,_=resolve_quality_perception(a.adaptation_run,a.checkpoint);review=EnvironmentReview(a.bundle,perception_identity=identity)
 if a.examples<1:raise ValueError('positive example count required')
 out.mkdir(parents=True,exist_ok=False);code=out/'code';code.mkdir();commit=git(repo,'rev-parse','HEAD');archive=out/'source_archive.tar'
 subprocess.run(['git','-C',str(repo),'archive','--format=tar','--output',str(archive),commit,*source_paths],check=True)
 with tarfile.open(archive) as tar:tar.extractall(code,filter='data')
 archive.unlink();upstream=repo/'third_party/TwinLiteNetPlus';revision=git(upstream,'rev-parse','HEAD')
 if git(upstream,'status','--porcelain','--untracked-files=no'):raise ValueError('upstream tracked checkout changed')
 target=code/'third_party/TwinLiteNetPlus';target.parent.mkdir(exist_ok=True)
 subprocess.run(['git','-c','core.logAllRefUpdates=false','clone','--quiet','--no-local','--no-checkout',str(upstream),str(target)],check=True)
 url=git(repo,'config','--file','.gitmodules','--get','submodule.third_party/TwinLiteNetPlus.url')
 subprocess.run(['git','-C',str(target),'remote','set-url','origin',url],check=True);subprocess.run(['git','-C',str(target),'-c','core.logAllRefUpdates=false','checkout','--quiet','--detach',revision],check=True)
 if git(target,'rev-parse','HEAD')!=revision or (target/'.git/objects/info/alternates').exists():raise ValueError('upstream checkout is not self-contained')
 subprocess.run(['git','-C',str(target),'fsck','--connectivity-only','--no-reflogs'],check=True,capture_output=True)
 if (target/'.git/objects/info/alternates').exists():raise ValueError('upstream clone depends on original workspace')
 if (target/'.git/logs').exists():shutil.rmtree(target/'.git/logs')
 def copy(src,relative):
  dest=(out/relative).resolve();dest.relative_to(out);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
  if digest(src)!=digest(dest):raise ValueError('copy checksum differs')
 if a.adaptation_run:
  run=json.loads((a.adaptation_run/'run.json').read_text())
  for name in ['run.json','_SUCCESS',*run['artifact_sha256']]:
   src=(a.adaptation_run/name).resolve();src.relative_to(a.adaptation_run.resolve());copy(src,'models/perception/'+name)
 else:copy(checkpoint,'models/perception/best_mean.pt')
 copy(a.bundle/'bundle_manifest.json','models/review/bundle_manifest.json')
 for name in review.spec['artifact_sha256']:copy(a.bundle/name,'models/review/'+name)
 rows=[json.loads(line) for line in a.rgb_manifest.read_text().splitlines() if line.strip()]
 if len(rows)<a.examples:raise ValueError('not enough independent RGB examples')
 step=max(1,(len(rows)-1)/max(a.examples-1,1));indices=[min(len(rows)-1,round(i*step)) for i in range(a.examples)]
 if len(set(indices))!=a.examples:raise ValueError('not enough independent RGB examples')
 examples=[]
 for i,index in enumerate(indices):
  row=rows[index];image=(a.rgb_root/row['image']).resolve();image.relative_to(a.rgb_root.resolve())
  if row.get('view')!='front':raise ValueError('fixed front camera examples required')
  if digest(image)!=row['expected_sha256']['rgb']:raise ValueError('source RGB example changed')
  name=f'{i:04d}'+image.suffix;copy(image,'examples/rgb/'+name);examples.append(dict(sample_id=f'demo_{i:04d}',image=name,view='front',expected_sha256={'rgb':digest(image)}))
 (out/'examples/manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in examples))
 argument='--adaptation-run /release/models/perception' if a.adaptation_run else '--checkpoint /release/models/perception/best_mean.pt'
 text=f'''# Q/environment RGB inference handoff

Source: `{commit}`. Offline candidate status: **{a.adoption_status}**.
This private team handoff includes exact training lineage metadata when required by the perception identity. It includes {len(examples)} RGB examples and no Semantic/GT image. A successful replay does not adopt a rejected offline candidate or connect live ROS.

Set RELEASE_ROOT to this extracted folder and REPLAY_OUTPUT to a new empty writable output folder. Build the appropriate GPU profile from the team repository first.

```bash
python3 "$RELEASE_ROOT/code/scripts/verify_environment_release.py" --root "$RELEASE_ROOT"
docker run --rm -v "$RELEASE_ROOT:/release:ro" -v "$REPLAY_OUTPUT:/output" \\
  -w /release/code -e PYTHONPATH=/release/code/src:/release/code/ros/src/data_collection/src \\
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 ml-perception:rtx40-cu118 \\
  python -B -m quality_estimation.infer_versioned_review \\
  --bundle /release/models/review {argument} \\
  --manifest /release/examples/manifest.jsonl --data-root /release/examples/rgb \\
  --output /output/replay --device cpu --batch-size 4 --budget {min(4,len(examples))}
```

For50-series GPU inference add `--gpus all`, select `ml-perception:rtx50-cu128`, and use `--device cuda`. Actual40-series GPU execution remains a teammate hardware check. The source checkout, upstream Git objects, model/calibration/policy hashes and exact perception lineage are self-contained.
'''
 (out/'README.md').write_text(text)
 files={str(path.relative_to(out)):digest(path) for path in sorted(out.rglob('*')) if path.is_file()};manifest=dict(version='environment-inference-release-1',created_utc=datetime.now(timezone.utc).isoformat(),source_git=commit,upstream_git=revision,perception_identity=identity,adoption_status=a.adoption_status,rgb_examples=len(examples),contains_raw_ground_truth=False,contains_training_lineage=a.adaptation_run is not None,files=files)
 (out/'release_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(verify(out),indent=2))
if __name__=='__main__':main()
