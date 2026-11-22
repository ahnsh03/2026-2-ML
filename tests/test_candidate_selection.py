"""Operational candidate admission needs RGB lineage and scores, never GT."""
import copy
import unittest
import numpy as np

from quality_estimation.select_candidates import plan_selection
from quality_estimation.perception_data import validate_adaptation_selection


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

    def test_diversity_and_uniform_draw_share_only_the_high_risk_subset(self):
        rows=[dict(sample_id=str(i),frame_id=i,run_id="r",preregistered_quality_role="candidate_pool") for i in range(8)]
        pred=[dict(sample_id=str(i),pixel_sha256=str(i),rgb_sha256=str(i),risk_score=1-i/8) for i in range(8)]
        X=np.zeros((8,56)); X[:,0]=[0,.1,.3,8,99,100,101,102]
        plan=plan_selection(rows,pred,X,[.25],[4],2)
        diverse,uniform=plan["selections"][-2:]
        self.assertEqual(diverse["indices"],[0,3])
        self.assertTrue(set(uniform["indices"]) <= set(range(4)))
        self.assertEqual(diverse["risk_pool_sha256"],uniform["risk_pool_sha256"])
        self.assertEqual(uniform["risk_pool_rows"],4)
        X[:]=0
        tied=plan_selection(rows,pred,X,[.25],[4],2)["selections"][-2]
        self.assertEqual(tied["indices"],[0,1])
        entries=[dict(selection=dict(uniform,rank=i+1,inference_run_sha256="a"*64,model_run_sha256="b"*64)) for i in range(2)]
        validate_adaptation_selection(entries,"RiskPoolRandom",42,.25)
        entries[0]["selection"]["risk_pool_sha256"]="d"*64
        with self.assertRaisesRegex(ValueError,"risk_pool_sha256"):
            validate_adaptation_selection(entries,"RiskPoolRandom",42,.25)

    def test_predicted_area_bands_fill_shared_quota_and_honor_boundaries(self):
        rows = [dict(sample_id=str(i), frame_id=i, run_id="r", preregistered_quality_role="candidate_pool") for i in range(6)]
        pred = [dict(sample_id=str(i), pixel_sha256=str(i), rgb_sha256=str(i), risk_score=1-i/6) for i in range(6)]
        X = np.zeros((6, 56), dtype=np.float32)
        X[:, 2] = [0, .001, .003, .01, .02, .5]
        risk, random = plan_selection(rows, pred, X, [.75], [4], area_stratified=True)["selections"][-2:]
        self.assertEqual(risk["indices"], [0, 1, 3, 2, 4])
        self.assertEqual(risk["area_pool_counts"], [1, 2, 3])
        self.assertEqual(risk["selected_area_bands"], random["selected_area_bands"])
        self.assertEqual(risk["area_pool_sha256"], random["area_pool_sha256"])
        entries = [dict(selection=dict(random, rank=i+1, predicted_area_band=band,
                        inference_run_sha256="a"*64, model_run_sha256="b"*64))
                   for i, band in enumerate(random["selected_area_bands"])]
        validate_adaptation_selection(entries, "AreaStratifiedRandom", 100, .75)
        entries[1]["selection"]["predicted_area_band"] = 2
        with self.assertRaisesRegex(ValueError, "area admission order"):
            validate_adaptation_selection(entries, "AreaStratifiedRandom", 100, .75)

    def test_area_stratified_budget_works_with_empty_bands(self):
        rows, pred, X = self.inputs()
        X[:, 2] = 0
        risk = plan_selection(rows, pred, X, [.34], [4], area_stratified=True)["selections"][-2]
        self.assertEqual(risk["selected_area_bands"], [0, 0])
        X[:, 2] = 1.2
        with self.assertRaisesRegex(ValueError, "predicted lane fraction"):
            plan_selection(rows, pred, X, [.34], [4], area_stratified=True)

    def test_reencoded_pixel_identity_is_required_and_score_ties_are_stable(self):
        rows,predictions,X=self.inputs()
        for item in predictions:item['risk_score']=.5
        plan=plan_selection(rows,predictions,X,[.34],[0])
        self.assertEqual(plan['selections'][0]['indices'],[0,1])
        del predictions[0]['pixel_sha256']
        with self.assertRaisesRegex(ValueError,'decoded-pixel hashes'):
            plan_selection(rows,predictions,X,[.1],[0])


if __name__=='__main__':unittest.main()
