"""Operational candidate admission needs RGB lineage and scores, never GT."""
import copy
import unittest
import numpy as np

from quality_estimation.select_candidates import plan_selection


class CandidateSelectionTests(unittest.TestCase):
    def inputs(self):
        rows=[dict(sample_id=str(i),run_id='run',parent_session_id='candidate',frame_id=i,
                   preregistered_quality_role='candidate_pool',image='missing_rgb.png',semantic='missing_GT.png') for i in range(4)]
        predictions=[dict(sample_id=str(i),pixel_sha256=pixel,rgb_sha256='file'+str(i),risk_score=risk)
                     for i,(pixel,risk) in enumerate(zip(('same','second','same','fourth'),(.9,.8,.99,.2)))]
        return rows,predictions,np.zeros((4,56),np.float32)

    def test_selection_ignores_annotations_and_counts_deduplicated_budget(self):
        rows,predictions,X=self.inputs();plan=plan_selection(rows,predictions,X,[.34],[20261116])
        self.assertEqual(plan['eligible'],[True,True,False,True])
        self.assertEqual(plan['pool_rows'],3)
        proposed=next(r for r in plan['selections'] if r['method']=='FrozenQualityRank')
        self.assertEqual(proposed['indices'],[0,1])
        self.assertEqual(proposed['selected_count'],2)  # ceil(3*0.34), without annotation filtering.
        changed=copy.deepcopy(rows)
        for i,row in enumerate(changed):row.update(semantic='different_missing_GT.png',lane_iou=i/4,label_defined=False)
        self.assertEqual(plan,plan_selection(changed,predictions,X,[.34],[20261116]))

    def test_final_pool_cannot_be_admitted_for_training(self):
        rows,predictions,X=self.inputs();rows[0]['preregistered_quality_role']='final_test'
        with self.assertRaisesRegex(ValueError,'reserved candidate'):
            plan_selection(rows,predictions,X,[.1],[0])

    def test_reencoded_pixel_identity_is_required_and_score_ties_are_stable(self):
        rows,predictions,X=self.inputs()
        for item in predictions:item['risk_score']=.5
        plan=plan_selection(rows,predictions,X,[.34],[0])
        self.assertEqual(plan['selections'][0]['indices'],[0,1])
        del predictions[0]['pixel_sha256']
        with self.assertRaisesRegex(ValueError,'decoded-pixel hashes'):
            plan_selection(rows,predictions,X,[.1],[0])


if __name__=='__main__':unittest.main()
