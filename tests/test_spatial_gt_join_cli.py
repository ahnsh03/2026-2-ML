import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from quality_estimation.join_spatial_gt import main, sha


class SpatialGTJoinCLITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.queue=self.root/'queue';self.packs=self.root/'packs'
        self.queue.mkdir();self.packs.mkdir();self.out=self.root/'joined'
        self.schema=self.packs/'schema.json'
        self.write(self.schema,dict(profile='roi9-front-v1',regions_sha256='d'*64))
        self.index={};self.candidates=[]
        for key in ['road_a','road_b']:
            rgb=self.root/(key+'_rgb');sem=self.root/(key+'_semantic')
            rgb.write_bytes(('rgb:'+key).encode());sem.write_bytes(('semantic:'+key).encode())
            self.index[key]=dict(rows=[dict(sample_id=key,image=rgb.name,semantic=sem.name,expected_sha256=dict(rgb=sha(rgb),semantic=sha(sem)))])
            counts=np.zeros((1,9,3),np.int64);counts[0,[4,6],0]=60
            np.savez_compressed(self.packs/(key+'.npz'),ids=np.array([key]),counts=counts,y=np.array([1.]))
            self.candidates.append(dict(sample_id=key,rgb_sha256=sha(rgb),checkpoint_sha256='c'*64,profile='roi9-front-v1',regions_sha256='d'*64))
        self.index_path=self.root/'index.json';self.write(self.index_path,self.index)
        for key in self.index:
            self.write(self.packs/(key+'.json'),dict(sha256=sha(self.packs/(key+'.npz')),schema_sha256=sha(self.schema),index_sha256=sha(self.index_path),checkpoint_sha256='c'*64))
        self.freeze_queue()

    @staticmethod
    def write(path,value):path.write_text(json.dumps(value))

    def freeze_queue(self):
        path=self.queue/'candidates.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in self.candidates))
        self.write(self.queue/'run.json',dict(status='complete',candidates_sha256=sha(path)))
        (self.queue/'_SUCCESS').write_text(sha(self.queue/'run.json'))

    def run_cli(self,*choice):
        args=['join_spatial_gt','--queue-dir',str(self.queue),'--packs',str(self.packs),'--source-index',str(self.index_path),'--data-root',str(self.root),'--output-dir',str(self.out),*choice]
        with patch('sys.argv',args),contextlib.redirect_stdout(io.StringIO()):main()

    def test_multi_road_queue_joins_verified_sources_without_admission(self):
        self.run_cli('--all-keys')
        rows=[json.loads(line) for line in (self.out/'gt_review.jsonl').read_text().splitlines()]
        self.assertEqual([r['sample_id'] for r in rows],['road_a','road_b'])
        self.assertTrue(all(r['true_lane_iou']==1 and not r['admission_approved'] for r in rows))
        record=json.loads((self.out/'run.json').read_text())
        self.assertEqual(len(record['source_packs_sha256']),2)
        self.assertTrue(record['selected_original_bytes_verified'])

    def test_mixed_perception_versions_are_rejected_before_output(self):
        p=self.packs/'road_b.json';r=json.loads(p.read_text());r['checkpoint_sha256']='e'*64;self.write(p,r)
        with self.assertRaisesRegex(ValueError,'different perception'):self.run_cli('--all-keys')
        self.assertFalse(self.out.exists())

    def test_changed_original_semantic_is_rejected_before_output(self):
        (self.root/'road_b_semantic').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum differs'):self.run_cli('--all-keys')
        self.assertFalse(self.out.exists())

    def test_legacy_single_key_keeps_single_pack_provenance(self):
        self.candidates=self.candidates[:1];self.freeze_queue();self.run_cli('--key','road_a')
        record=json.loads((self.out/'run.json').read_text())
        self.assertEqual(record['rows'],1)
        self.assertEqual(record['source_pack_sha256'],sha(self.packs/'road_a.npz'))


if __name__=='__main__':unittest.main()
