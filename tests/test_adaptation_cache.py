import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from quality_estimation.adaptation_cache import prepare_one,CachedAdaptationSamples
from quality_estimation.perception_data import NativePerceptionSamples
from data_collection.mask_baker import OFFICIAL_PALETTE_RGB

class AdaptationCacheTest(unittest.TestCase):
    def test_unknown_semantic_color_is_rejected_before_cache_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cv2.imwrite(str(root/'rgb.png'),np.full((36,64,3),127,np.uint8))
            cv2.imwrite(str(root/'sem.png'),np.full((36,64,3),1,np.uint8))
            with self.assertRaisesRegex(ValueError,'unknown Semantic'):
                prepare_one(root,dict(sample_id='bad',image='rgb.png',semantic='sem.png'),root/'cache')
    def test_cached_tensors_match_native_and_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'cache'
            bgr=np.random.default_rng(12).integers(0,256,(36,64,3),dtype=np.uint8)
            palette=[v[::-1] for k,v in OFFICIAL_PALETTE_RGB.items() if k!='blue_lane']
            semantic=np.empty_like(bgr)
            for j in range(semantic.shape[1]):semantic[:,j]=palette[j%len(palette)]
            cv2.imwrite(str(root/'rgb.png'),bgr);cv2.imwrite(str(root/'sem.png'),semantic)
            row=dict(sample_id='fixture',view='front',image='rgb.png',semantic='sem.png')
            cached=prepare_one(root,row,cache);actual=CachedAdaptationSamples(cache,[cached],verify=True)[0]
            expected=NativePerceptionSamples(root,[row])[0]
            np.testing.assert_array_equal(actual['image'].numpy(),expected['image'])
            for name in ['drivable','road_marking']:np.testing.assert_array_equal(actual['targets'][name].numpy(),expected['targets'][name])
            np.testing.assert_array_equal(actual['valid_mask'].numpy(),expected['valid_mask'])
            escaped = dict(cached, cache_relative='../outside')
            with self.assertRaises(ValueError):CachedAdaptationSamples(cache,[escaped],verify=True)
            many = CachedAdaptationSamples(cache,[cached]*17,verify=True)
            self.assertEqual(len(many),17)
            path=cache/cached['cache_relative']/'targets.png';path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError,'cache tensor changed'):CachedAdaptationSamples(cache,[cached],verify=True)

if __name__=='__main__':unittest.main()
