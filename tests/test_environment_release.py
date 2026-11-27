"""The handoff file contract must preserve lineage and exclude example GT."""
from pathlib import Path
import json,tempfile,unittest,sys

scripts=Path(__file__).resolve().parents[1]/'scripts'
if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
from verify_environment_release import verify
from verify_inference_release import digest


class EnvironmentReleaseTest(unittest.TestCase):
 def fixture(self,root):
  def write(name,value):
   p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value) if isinstance(value,(dict,list)) else value);return p
  checkpoint=write('models/perception/checkpoint.pt','test-only checkpoint bytes');write('models/perception/protocol.json',{});write('models/perception/training_membership.json',[])
  artifacts={n:digest(root/'models/perception'/n) for n in ['checkpoint.pt','protocol.json','training_membership.json']};run=write('models/perception/run.json',dict(status='complete',artifact_sha256=artifacts));write('models/perception/_SUCCESS',digest(run)+'\n')
  identity=dict(kind='adaptation',checkpoint_sha256=digest(checkpoint),adaptation_run_sha256=digest(run));model=write('models/review/quality_model.npz','test-only model bytes');write('models/review/bundle_manifest.json',dict(perception_identity=identity,artifact_sha256={'quality_model.npz':digest(model)}))
  rgb=write('examples/rgb/0000.png','test-only RGB bytes');write('examples/manifest.jsonl',json.dumps(dict(sample_id='demo',image='0000.png',view='front',expected_sha256={'rgb':digest(rgb)}))+'\n')
  spec=dict(version='environment-inference-release-1',source_git='1'*40,perception_identity=identity,rgb_examples=1,files={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()});write('release_manifest.json',spec);return spec
 def test_preserves_exact_adaptation_and_rgb_only_context(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);self.fixture(root);r=verify(root);self.assertEqual(r['rgb_examples'],1);self.assertTrue(r['contains_training_lineage']);self.assertFalse(r['contains_raw_ground_truth'])
 def test_rejects_changed_weights_escape_and_gt_metadata(self):
  for mode in ['weights','escape','gt','view']:
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)/'release';root.mkdir();spec=self.fixture(root)
    if mode=='weights':(root/'models/perception/checkpoint.pt').write_text('changed')
    elif mode=='escape':
     outside=root.parent/'outside';outside.write_text('not a release artifact');spec['files']['../outside']=digest(outside)
    else:
     path=root/'examples/manifest.jsonl';row=json.loads(path.read_text());row.update({'semantic':'not-allowed.png'} if mode=='gt' else {'view':'left'});path.write_text(json.dumps(row)+'\n');spec['files']['examples/manifest.jsonl']=digest(path)
    (root/'release_manifest.json').write_text(json.dumps(spec))
    with self.assertRaises(ValueError):verify(root)
if __name__=='__main__':unittest.main()
