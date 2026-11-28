import itertools
import unittest

import numpy as np

from quality_estimation.cluster_comparison import environment_pairs, assign_centers, fit_centers
from quality_estimation.rf_search import rf_space, evaluate_rf
from quality_estimation.roi9_review import score_rows, replay_policy, low_quality_mask


class RFComparisonTests(unittest.TestCase):
    def test_rf_space_preserves_baseline_and_normalized_coordinates(self):
        spec = dict(trees=8, min_samples_leaf=[1,2,4], max_features=[.5,1.], max_depth=[8,None],
                    baseline=dict(n_estimators=8,min_samples_leaf=2,max_features=.5,max_depth=None))
        points, coordinates, base = rf_space(spec)
        self.assertEqual(len(points), 12)
        self.assertEqual(points[base], spec["baseline"])
        self.assertTrue(((coordinates>=0)&(coordinates<=1)).all())
        self.assertEqual(len(np.unique(coordinates,axis=0)),len(points))

    def test_rf_cv_predictions_have_one_validation_prediction_per_row(self):
        x = np.arange(12,dtype=np.float32)[:,None]
        y = x[:,0]/12
        folds = [(str(k),np.flatnonzero(np.arange(12)%3!=k),np.flatnonzero(np.arange(12)%3==k)) for k in range(3)]
        score,pred = evaluate_rf(x,y,folds,dict(n_estimators=8,min_samples_leaf=1,max_features=1.,max_depth=None),7)
        self.assertTrue(np.isfinite(pred).all())
        self.assertAlmostEqual(score["cv_mae"],float(np.mean(abs(pred-y))),places=6)


class ClusterComparisonTests(unittest.TestCase):
    def test_pair_counts_match_explicit_distinct_image_pairs(self):
        labels = np.array([0,0,1,1,2,0])
        environments = np.array([0,0,1,1,0,1])
        sites = np.array([0,1,0,1,2,0])
        a = [(i,j) for i,j in itertools.combinations(range(len(labels)),2) if environments[i]==environments[j] and sites[i]!=sites[j]]
        b = [(i,j) for i,j in itertools.combinations(range(len(labels)),2) if sites[i]==sites[j] and environments[i]!=environments[j]]
        result = environment_pairs(labels,environments,sites)
        self.assertEqual(result["same_environment_different_site_pairs"],len(a))
        self.assertEqual(result["same_site_changed_environment_pairs"],len(b))
        self.assertAlmostEqual(result["same_environment_cluster_agreement"],np.mean([labels[i]==labels[j] for i,j in a]))
        self.assertAlmostEqual(result["changed_environment_cluster_separation"],np.mean([labels[i]!=labels[j] for i,j in b]))

    def test_missing_pair_denominator_is_not_reported_as_zero_accuracy(self):
        result = environment_pairs([0,1],[0,1],[0,1])
        self.assertIsNone(result["same_environment_cluster_agreement"])
        self.assertIsNone(result["changed_environment_cluster_separation"])

    def test_final_centers_are_reassigned_and_input_initialization_is_preserved(self):
        x=np.array([[0.,0.],[0.,1.],[9.,9.],[10.,9.]],dtype=np.float32)
        initial=x[[0,2]].copy()
        centers,details=fit_centers("kmeans",x,initial,0,5)
        labels,mse=assign_centers(x,centers)
        np.testing.assert_array_equal(initial,x[[0,2]])
        np.testing.assert_array_equal(labels,[0,0,1,1])
        self.assertAlmostEqual(mse,.25)
        self.assertLessEqual(details["lloyd_iterations"],5)


class ReviewComparisonTests(unittest.TestCase):
    def test_float32_threshold_equality_is_normal_in_all_aggregations(self):
        y=np.array([.69,.7,.71],dtype=np.float32)
        np.testing.assert_array_equal(low_quality_mask(y),[True,False,False])

    def inputs(self):
        rows=[dict(sample_id=f"f{i:02}",pixel_sha256=format(i,"064x"),run_id="run",drive_order=str(i),true_iou=12345) for i in range(20)]
        return rows,np.linspace(.9,.1,20),np.linspace(.1,.9,20),np.linspace(.01,.5,20),np.arange(20)%3

    def test_sigma_mixture_preserves_audit_and_uses_only_whitelisted_scores(self):
        rows,audit,priority=score_rows(*self.inputs(),"gp_sigma_audit")
        self.assertNotIn("true_iou",rows[0])
        selected,population,_=replay_policy(rows,"static",50,7,audit,priority)
        reasons=[reason for _,reason in selected]
        self.assertEqual(len(selected),10)
        self.assertEqual(reasons.count("random_remainder_audit"),2)
        self.assertEqual(reasons.count("low_estimated_iou"),6)
        self.assertEqual(reasons.count("high_prediction_std"),2)

    def test_all_policies_have_same_chronological_budget_and_population(self):
        results=[]
        for policy in ["random","rf_mean","gp_mean","gp_mean_audit","gp_sigma_audit","gp_sigma_only","gp_lcb","gp_environment"]:
            rows,audit,priority=score_rows(*self.inputs(),policy)
            selected,population,windows=replay_policy(rows,"windows",10,7,audit,priority,8)
            results.append((len(selected),[r["sample_id"] for r in population],[w["budget"] for w in windows]))
        self.assertTrue(all(r==results[0] for r in results))
        self.assertEqual(results[0][0],2)

    def test_static_selection_is_order_independent_and_pixel_deduplicated(self):
        rows,audit,priority=score_rows(*self.inputs(),"gp_mean_audit")
        rows.append(dict(rows[0],sample_id="duplicate"))
        first,pool,_=replay_policy(rows,"static",20,7,audit,priority)
        second,_,_=replay_policy(rows[::-1],"static",20,7,audit,priority)
        self.assertEqual(len(pool),20)
        self.assertEqual(first,second)
        self.assertEqual(len(first),4)

    def test_ambiguous_chronology_is_rejected(self):
        rows,audit,priority=score_rows(*self.inputs(),"gp_mean")
        rows[0]["drive_order"]=""
        with self.assertRaisesRegex(ValueError,"drive order"):
            replay_policy(rows,"windows",10,7,audit,priority)


if __name__ == "__main__":
    unittest.main()
