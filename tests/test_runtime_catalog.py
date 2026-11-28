import hashlib,json,tempfile,unittest
from pathlib import Path
from quality_estimation.runtime_catalog import verify_repository_bundle

class RuntimeCatalogTest(unittest.TestCase):
    def test_changed_manifest_requires_a_new_registered_release(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);bundle=root/'models/review/reference';bundle.mkdir(parents=True);manifest=bundle/'bundle_manifest.json';manifest.write_text('{}')
            (root/'models/catalog.json').write_text(json.dumps(dict(reference_review_bundle='models/review/reference',review_bundle_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())))
            verify_repository_bundle(bundle,root);manifest.write_text('{"changed":true}')
            with self.assertRaises(ValueError):verify_repository_bundle(bundle,root)
    def test_unregistered_repository_model_is_not_auto_used(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);(root/'models').mkdir();(root/'models/catalog.json').write_text(json.dumps(dict(reference_review_bundle='models/review/reference',review_bundle_manifest_sha256='a'*64)))
            with self.assertRaises(ValueError):verify_repository_bundle(root/'models/review/unregistered',root)
