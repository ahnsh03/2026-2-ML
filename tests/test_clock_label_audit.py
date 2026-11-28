import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/experiments/scene_cycle_20261128/audit_clock_labels.py'
spec=importlib.util.spec_from_file_location('clock_label_audit',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ClockAuditTest(unittest.TestCase):
    def document(self):
        def counts(errors,picked):
            return dict(rows=20,selected_rows=picked,near_error=errors,near_error_selected=picked,
                near_fn=errors,near_fn_selected=picked,near_fp=0,near_fp_selected=0)
        return dict(results=dict(primary=dict(support_by_hour={'5':counts(10,5),'7':counts(2,0)},review=dict(dawn_near_error_recall=0.))))

    def test_05_and_07_remain_separate_and_no_fp_is_undefined(self):
        result=m.audit(self.document())['results']['primary']
        self.assertEqual(result['sdk_hours']['5']['near_error_recall'],.5)
        self.assertEqual(result['sdk_hours']['7']['near_error_recall'],0.)
        self.assertIsNone(result['sdk_hours']['5']['near_fp_recall'])

    def test_relabeling_05_as_legacy07_is_rejected(self):
        document=self.document();document['results']['primary']['review']['dawn_near_error_recall']=.5
        with self.assertRaises(ValueError):m.audit(document)
