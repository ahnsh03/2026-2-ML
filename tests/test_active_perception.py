import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from quality_estimation.active_perception import resolve_active_perception


class ActivePerceptionTest(unittest.TestCase):
    def fixture(self,path):
        identity=dict(checkpoint_sha256='a'*64,adaptation_run_sha256='b'*64)
        path.write_text(json.dumps(dict(schema_version='perception-development-releases-1',
            active_development_version='r1',releases={'r1':dict(kind='adaptation',**identity)})))
        return identity

    def test_matching_release_and_lineage_are_resolved_without_adopting_policy(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'registry.json';identity=self.fixture(path)
            with patch('quality_estimation.active_perception.resolve_quality_perception',return_value=(Path('checkpoint'),identity,[{}])):
                result=resolve_active_perception(path,'run')
                self.assertEqual(result[0],'r1');self.assertEqual(result[2],identity)

    def test_wrong_model_or_lineage_rejected(self):
        for field in ['checkpoint_sha256','adaptation_run_sha256']:
            with self.subTest(field=field),tempfile.TemporaryDirectory()as tmp:
                path=Path(tmp)/'registry.json';identity=self.fixture(path);identity[field]='c'*64
                with patch('quality_estimation.active_perception.resolve_quality_perception',return_value=(Path('checkpoint'),identity,[])):
                    with self.assertRaises(ValueError):resolve_active_perception(path,'run')
