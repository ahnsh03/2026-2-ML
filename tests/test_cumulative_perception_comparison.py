import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/experiments/scene_cycle_20261128/compare_perception.py'
spec=importlib.util.spec_from_file_location('cumulative_perception_comparison',path)
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


class CumulativePerceptionComparisonTests(unittest.TestCase):
    def row(self,site,lane,role='development',view='front'):
        return dict(site=site,adaptation_role=role,view=view,
                    counts={head:lane for head in ['lane','drivable','white','yellow','stopline']})

    def test_marked_segment_macro_does_not_reward_longer_capture(self):
        rows=[self.row('a',[90,0,10])for _ in range(10)]+[self.row('b',[10,0,90])]
        rows += [self.row('old',[50,10,40],role='retention',view=v)for v in ['front','left','right']]
        result=comparison.metrics(rows,['a','b'])
        self.assertAlmostEqual(result['marked_macro'],.5)
        self.assertAlmostEqual(result['retention']['lane'],.5)
        self.assertAlmostEqual(result['retention_by_view']['left']['drivable'],.5)

    def test_false_positive_only_segment_cannot_supply_marked_gt_support(self):
        rows=[self.row('a',[0,100,0])]
        with self.assertRaisesRegex(ValueError,'lacks supported GT'):comparison.metrics(rows,['a'])


if __name__=='__main__':unittest.main()
