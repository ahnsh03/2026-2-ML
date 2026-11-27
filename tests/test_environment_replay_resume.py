from pathlib import Path
from types import SimpleNamespace
import json,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/experiments/environment_cycle_20261127'))
from check_runtime import verify_cached_replay
from study_data import sha


class EnvironmentReplayResumeTest(unittest.TestCase):
 def test_policy_change_cannot_reuse_an_old_complete_replay(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);out=root/'out';out.mkdir();bundle=root/'bundle';bundle.mkdir();manifest=root/'input.jsonl';manifest.write_text('{}\n');bm=bundle/'bundle_manifest.json';bm.write_text('{"policy": "old"}\n');artifact=out/'predictions.jsonl';artifact.write_text('{}\n');identity={'checkpoint_sha256':'a'*64};review=SimpleNamespace(root=bundle,spec={'perception_identity':identity})
   meta=dict(status='complete',bundle_manifest_sha256=sha(bm),manifest_sha256=sha(manifest),perception_identity=identity,artifact_sha256={'predictions.jsonl':sha(artifact)});(out/'run.json').write_text(json.dumps(meta));(out/'_SUCCESS').write_text(sha(out/'run.json')+'\n');verify_cached_replay(out,review,manifest)
   bm.write_text('{"policy": "new, same perception checkpoint"}\n')
   with self.assertRaisesRegex(ValueError,'another bundle'):verify_cached_replay(out,review,manifest)
