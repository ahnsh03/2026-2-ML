"""Verify annotation-free inference and reject changed model/split artifacts."""
import json
import tempfile
from pathlib import Path
import unittest

import cv2
import joblib
import numpy as np

from common.bridge import DEFAULT_CHECKPOINT_SHA256, sha256_file
from quality_estimation.features import FEATURE_NAMES, feature_schema
from quality_estimation.infer_images import RGBSamples
from quality_estimation.model_io import load_model_run
from quality_estimation.models import make_estimator, predict_array
from quality_estimation.dataset_io import QualityDataset
from quality_estimation.evaluate_models import reject_training_overlap
from quality_estimation.freeze_models import create_freeze
from quality_estimation.splits import build_split, read_split


class InferenceTests(unittest.TestCase):
    def test_final_test_rejects_renamed_selection_pixels_and_parent_reuse(self):
        bundle=dict(train_parent_groups=['train'],selection_parent_groups=['dev'])
        membership=[dict(sample_id='fit',pixel_sha256='fitpixels',parent_session_id='train')]
        selected=[dict(sample_id='tune',pixel_sha256='devpixels',parent_session_id='dev')]
        sample=dict(sample_id='renamed',pixel_sha256='devpixels',parent_session_id='final',input_sha256={'rgb':'different_file_encoding'})
        with self.assertRaisesRegex(ValueError,'pixels'):
            reject_training_overlap(bundle,membership,[sample],True,selected)
        with self.assertRaisesRegex(ValueError,'pixels'):
            reject_training_overlap(bundle,membership,[sample],selection_membership=selected)
        sample.update(pixel_sha256='newpixels',parent_session_id='dev')
        with self.assertRaisesRegex(ValueError,'parent'):
            reject_training_overlap(bundle,membership,[sample],True,selected)
        sample['parent_session_id']='final'
        reject_training_overlap(bundle,membership,[sample],True,selected)

    def test_final_test_requires_selection_lineage(self):
        with self.assertRaisesRegex(ValueError,'selection membership'):
            reject_training_overlap(dict(train_parent_groups=[],selection_parent_groups=[]),[],[],True)

    def test_rgb_loader_never_opens_annotation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rgb = np.zeros((36,64,3),np.uint8); rgb[:,:,2] = 160
            cv2.imwrite(str(root/'rgb.png'),rgb)
            row = dict(sample_id='one',view='front',image='rgb.png',semantic='missing.png',y='not a label')
            first = RGBSamples(root,[row])[0]
            row['semantic']='another/missing.png'; row['y']=123
            second = RGBSamples(root,[row])[0]
            np.testing.assert_array_equal(first['image'],second['image'])
            self.assertNotIn('target',first)
            self.assertEqual(first['rgb'][0,0].tolist(),[160,0,0])

    def _model_run(self,root):
        (root/'models').mkdir()
        (root/'protocol.json').write_text('{}')
        (root/'comparison.json').write_text('{}')
        (root/'train_membership.json').write_text('[]')
        X=np.zeros((4,56)); X[:,0]=[0,1,2,3]
        estimator=make_estimator('Ridge',{'alpha':1.},0).fit(X,[0,.2,.4,.6])
        bundle=dict(model='Ridge',estimator=estimator,checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,
                    feature_schema=feature_schema(),ordered_features=list(FEATURE_NAMES),
                    column_indices=list(range(56)),feature_set='full',
                    protocol_sha256=sha256_file(root/'protocol.json'))
        joblib.dump(bundle,root/'models/Ridge.joblib')
        paths=['models/Ridge.joblib','protocol.json','comparison.json','train_membership.json']
        (root/'run.json').write_text(json.dumps(dict(status='complete',selected_model='Ridge',
            artifact_sha256={name:sha256_file(root/name) for name in paths})))
        (root/'_SUCCESS').write_text(sha256_file(root/'run.json')+'\n')
        return bundle

    def test_saved_bundle_roundtrip_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); original=self._model_run(root)
            loaded,_=load_model_run(root)
            X=np.zeros((2,56)); X[:,0]=[1,2]
            np.testing.assert_array_equal(predict_array(original,X)['estimated_iou'],
                                          predict_array(loaded,X)['estimated_iou'])
            (root/'protocol.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                load_model_run(root)

    def test_incomplete_model_run_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'run.json').write_text('{"status":"running"}')
            with self.assertRaisesRegex(ValueError,'not complete'):
                load_model_run(root)

    def test_freeze_needs_selection_history_and_rejects_reused_group(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bundle=self._model_run(root)
            with self.assertRaisesRegex(ValueError,'selection membership'):
                create_freeze(root,['final'])
            bundle.update(train_parent_groups=['train'],selection_parent_groups=['dev'])
            joblib.dump(bundle,root/'models/Ridge.joblib')
            (root/'selection_membership.json').write_text('[]')
            run=json.loads((root/'run.json').read_text())
            run['artifact_sha256']['models/Ridge.joblib']=sha256_file(root/'models/Ridge.joblib')
            run['artifact_sha256']['selection_membership.json']=sha256_file(root/'selection_membership.json')
            (root/'run.json').write_text(json.dumps(run))
            (root/'_SUCCESS').write_text(sha256_file(root/'run.json')+'\n')
            with self.assertRaisesRegex(ValueError,'used for training or selection'):
                create_freeze(root,['dev'])
            frozen=create_freeze(root,['final'])
            self.assertEqual(frozen['evaluation_parent_groups'],['final'])
            self.assertFalse(frozen['final_test_accessed_before_freeze'])
            # No final dataset or annotation file exists in this test.
            (root/'schema.json').write_text(json.dumps(feature_schema(),ensure_ascii=False,indent=2)+'\n')
            self.assertEqual(frozen['feature_schema_sha256'],sha256_file(root/'schema.json'))

    def test_split_reader_rejects_changed_group_map(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'dataset.json').write_text('{}')
            samples=[dict(sample_id=str(i),run_id=str(i),parent_session_id=str(i),source_role='after_f0_training',
                          input_sha256={'rgb':str(i)}) for i in range(2)]
            data=QualityDataset(root,np.zeros((2,56)),np.zeros(2),np.ones(2,bool),['0','1'],samples,{},list(FEATURE_NAMES))
            split=build_split(data,{'0':'meta_train','1':'meta_dev'})
            split['group_roles']['1']='final_test'
            path=root/'split.json';path.write_text(json.dumps(split))
            with self.assertRaisesRegex(ValueError,'group map'):
                read_split(path,data)


if __name__=='__main__':
    unittest.main()
