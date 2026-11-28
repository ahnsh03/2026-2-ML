"""Apply the frozen perception/review fallback hierarchy, never select a test winner."""
from pathlib import Path
import argparse
import json
from common.bridge import sha256_file


def read(path):return json.loads(Path(path).read_text())


def decide(freeze,neural,review):
    parent=freeze['perception_models']['r1']['perception_identity']
    selected=freeze['perception_models']['selected']['perception_identity']
    unchanged=parent==selected
    if neural['selected_is_existing_parent']!=unchanged:
        raise ValueError('neural decision and frozen parent identity differ')
    if neural['new_child_passes']and(unchanged or neural['failed_gates']):
        raise ValueError('inconsistent child acceptance report')
    child=bool(neural['new_child_passes'])
    name=freeze['primary_review_name']if child else freeze['parent_review_name']
    identity=selected if child else parent
    candidate=next(c for c in freeze['review_candidates']if c['name']==name)
    result=review['results'][name]
    if(candidate['perception_identity']!=identity or result['perception_identity']!=identity
            or result['bundle_manifest_sha256']!=candidate['bundle_manifest_sha256']):
        raise ValueError('fallback review policy belongs to another neural version')
    quality=result.get('quality_handoff_gate')
    review_passes=bool(result['review_gate']['passes']and quality and quality['passes'])
    reasons=[]
    if not child and not unchanged:reasons.extend('child:'+gate for gate in neural['failed_gates'])
    reasons.extend('review:'+gate for gate in result['review_gate']['failed_gates'])
    if quality is None:reasons.append('quality:missing_handoff_guard')
    else:reasons.extend('quality:'+gate for gate in quality['failed_gates'])
    return dict(perception_decision='accept_frozen_child'if child else'retain_promoted_parent',
        perception_version='m-v6-r2-20261128'if child else'm-v6-r1-20261128',
        perception_identity=identity,review_name=name,review_bundle=candidate['bundle'],
        review_bundle_manifest_sha256=candidate['bundle_manifest_sha256'],
        offline_review_criteria_pass=review_passes,reasons=reasons,
        release_status='offline_criteria_pass_pending_runtime_handoff'if review_passes else'perception_only_review_not_accepted',
        fresh_confirmation_consumed=True,post_test_control_substitution=False,
        automatic_training_admission=False,live_connection_completed=False,
        scope='Only the frozen primary or its prospectively declared exact-parent fallback may be accepted. Raw training, family, uncertainty and environment controls never replace it after confirmation. Standalone runtime validation and release registration remain separate.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['freeze','neural-report','review-assessment','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();freeze=read(a.freeze);neural=read(a.neural_report);review=read(a.review_assessment)
    if neural['source_freeze_sha256']!=sha256_file(a.freeze)or review['confirmation_freeze_sha256']!=sha256_file(a.freeze):
        raise ValueError('adoption reports use different freezes')
    if neural['source_index_sha256']!=review['source_index_sha256']:
        raise ValueError('neural/review confirmation populations differ')
    if a.output.exists():raise ValueError('preserve existing adoption decision')
    result=decide(freeze,neural,review)
    result.update(version='frozen-offline-adoption-decision-1',
        confirmation_freeze_sha256=sha256_file(a.freeze),neural_report_sha256=sha256_file(a.neural_report),
        review_assessment_sha256=sha256_file(a.review_assessment),source_sha256=sha256_file(Path(__file__)),
        selected_perception_run=freeze['perception_models']['selected'if result['perception_decision']=='accept_frozen_child'else'r1']['run'])
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('OFFLINE DECISION',result['perception_decision'],result['review_name'],result['release_status'],flush=True)


if __name__=='__main__':main()
