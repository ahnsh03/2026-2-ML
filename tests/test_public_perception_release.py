import hashlib,json,tempfile,unittest
from pathlib import Path
from quality_estimation.perception_identity import baseline_identity,resolve_public_perception,resolve_quality_perception


class PublicPerceptionReleaseTest(unittest.TestCase):
    def fixture(self,root):
        model=root/'model';model.mkdir();checkpoint=model/'checkpoint.pt';checkpoint.write_bytes(b'verified-test-model-bytes')
        identity=dict(baseline_identity(),kind='adaptation',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),adaptation_run_sha256='b'*64)
        manifest=model/'release.json';manifest.write_text(json.dumps(dict(schema_version='public-perception-release-1',version='test-r2',checkpoint='checkpoint.pt',inference_only=True,perception_weights='model_state_dict',perception_identity=identity)))
        registry=root/'registry.json';registry.write_text(json.dumps(dict(schema_version='perception-development-releases-1',releases={'test-r2':dict(kind='adaptation',checkpoint_sha256=identity['checkpoint_sha256'],adaptation_run_sha256=identity['adaptation_run_sha256'],repository_release_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())})))
        return model,registry,identity

    def test_registered_exact_bytes_preserve_original_identity_without_training_ledger(self):
        with tempfile.TemporaryDirectory()as temp:
            model,registry,identity=self.fixture(Path(temp));path,actual,rows=resolve_public_perception(model,registry)
            self.assertEqual(path,model/'checkpoint.pt');self.assertEqual(actual,identity);self.assertEqual(rows,[])

    def test_tampered_checkpoint_or_manifest_is_rejected(self):
        for filename in ['checkpoint.pt','release.json']:
            with self.subTest(filename=filename),tempfile.TemporaryDirectory()as temp:
                model,registry,_=self.fixture(Path(temp));p=model/filename;p.write_bytes(p.read_bytes()+b' ')
                with self.assertRaises(ValueError):resolve_public_perception(model,registry)

    def test_checkpoint_escape_rejected_even_if_manifest_is_re_pinned(self):
        with tempfile.TemporaryDirectory()as temp:
            model,registry,_=self.fixture(Path(temp));manifest=model/'release.json';data=json.loads(manifest.read_text());data['checkpoint']='../checkpoint.pt';manifest.write_text(json.dumps(data));catalog=json.loads(registry.read_text());catalog['releases']['test-r2']['repository_release_manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest();registry.write_text(json.dumps(catalog))
            with self.assertRaises(ValueError):resolve_public_perception(model,registry)

    def test_deployment_and_training_sources_cannot_mix(self):
        with self.assertRaises(ValueError):resolve_quality_perception('training-run',release='deployment')
