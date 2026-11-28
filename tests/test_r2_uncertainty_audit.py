import importlib.util
from pathlib import Path

import numpy as np


spec=importlib.util.spec_from_file_location(
    'audit_r2',Path(__file__).resolve().parents[1]/'scripts/audit_reproduced_r2_uncertainty.py')
module=importlib.util.module_from_spec(spec)
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
spec.loader.exec_module(module)


def test_segment_macro_keeps_unequal_segment_sizes_and_coverage():
    pred=dict(estimated_iou=np.array([0.,0.,0.,1.]),
              interval90_lower=np.array([0.,0.,0.,.8]),
              interval90_upper=np.array([.2,.2,.2,1.]))
    result=module.quality_metrics(pred,np.zeros(4),np.array(['a','a','a','b']),np.ones(4,bool))
    assert result['mae']==.25
    assert result['macro_mae']==.5
    assert result['coverage']==.75
    assert result['macro_coverage']==.5


def test_population_filter_excludes_unsupported_and_undefined_truth():
    pred=dict(estimated_iou=np.array([.5,0.,0.]),
              interval90_lower=np.zeros(3),interval90_upper=np.ones(3))
    result=module.quality_metrics(pred,np.array([.5,1.,np.nan]),
                                  np.array(['a','b','c']),np.array([True,False,True]))
    assert result['rows']==1
    assert result['mae']==0
    assert result['coverage']==1
