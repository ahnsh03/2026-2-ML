"""Disambiguate SDK clock groups without changing frozen metrics or decisions."""
from pathlib import Path
import argparse
import json
from common.bridge import sha256_file


def audit(document):
    results={}
    for name,result in document['results'].items():
        hours={}
        for hour,counts in result['support_by_hour'].items():
            def recall(event):
                return counts[event+'_selected']/counts[event] if counts[event] else None
            hours[hour]=dict(rows=counts['rows'],selected=counts['selected_rows'],
                near_errors=counts['near_error'],near_errors_selected=counts['near_error_selected'],
                near_error_recall=recall('near_error'),near_fn_recall=recall('near_fn'),near_fp_recall=recall('near_fp'))
        legacy=result['review'].get('dawn_near_error_recall')
        expected=hours.get('7',{}).get('near_error_recall')
        if legacy is None or expected is None:
            if legacy is not expected:raise ValueError('Legacy07h definition differs')
        elif abs(legacy-expected)>1e-12:raise ValueError('Legacy07h definition differs')
        results[name]=dict(sdk_hours=hours,legacy_dawn_near_error_recall=legacy,
            legacy_dawn_metric_sdk_hour=7,legacy_definition_verified=True)
    return dict(version='explicit-sdk-clock-audit-1',results=results,
        acceptance_gates_or_selection_changed=False,
        scope='Post-assessment label clarification. The legacy dawn field denotes SDK07h only; SDK05h is reported separately. Clock hour is not a physical visibility label. All original reports and frozen decisions remain unchanged.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-assessment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous clock audit')
    result=audit(json.loads(a.review_assessment.read_text()))
    result.update(review_assessment_sha256=sha256_file(a.review_assessment),source_sha256=sha256_file(Path(__file__)))
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Explicit SDK clock labels verified')


if __name__=='__main__':main()
