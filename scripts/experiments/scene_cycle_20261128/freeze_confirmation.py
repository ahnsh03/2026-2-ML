"""Freeze primary, family controls and parent before collecting confirmation."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import subprocess
import sys
from common.bridge import sha256_file
from quality_estimation.perception_identity import resolve_quality_perception


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['study','models-ready','gpu-check','parent-run','baseline-checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--quality-guard',type=Path,help='Explicit prospectively recorded quality-denominator refinement')
    a=p.parse_args();s=a.study;ready=read(a.models_ready);gpu=read(a.gpu_check);repo=Path(__file__).resolve().parents[3]
    if(a.output.exists()or(s/'confirmation/COLLECTION_COMPLETE.json').exists()or list((s/'confirmation').glob('*_before.json'))
            or list((s/'confirmation/runs').glob('*/manifest.jsonl'))or list((s/'confirmation/captures').rglob('*_Intensity.png'))):raise ValueError('confirmation already frozen or collection attempted')
    if gpu['status']!='verified'or not gpu['original_retention_passes']or not gpu['development_passes']or gpu['selected_perception_identity']!=ready['perception_identity']:
        raise ValueError('selected perception lacks paired GPU retention verification')
    if(gpu['cpu_choice_sha256']!=sha256_file(s/'protocol/perception_development_choice.json')
            or gpu['study_design_sha256']!=sha256_file(s/'protocol/study_design_frozen.json')):raise ValueError('GPU verification belongs to another selection/design')
    source_paths=['src','config','docker','scripts','.gitmodules','ros/src/data_collection/src/data_collection']
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--',*source_paths],text=True).strip():raise ValueError('commit runtime/source before confirmation freeze')
    _,identity,_=resolve_quality_perception(Path(ready['perception_run']),None)
    if identity!=ready['perception_identity']:raise ValueError('selected perception lineage changed')
    _,parent_identity,_=resolve_quality_perception(a.parent_run,None);_,base_identity,_=resolve_quality_perception(None,a.baseline_checkpoint)
    selection=read(Path(ready['selection'])/'selection.json');models=Path(ready['models']);cal=Path(ready['calibration_pack']);envfit=Path(ready['environment_fit'])
    selected_pack=s/'confirmation/packs_selected/confirmation.npz';parent_pack=selected_pack if identity==parent_identity else s/'confirmation/packs_parent/confirmation.npz'
    for path in [selected_pack,parent_pack]:
        if path.exists():raise ValueError('confirmation features already inspected before freeze')
    root=s/'frozen_confirmation_bundles';root.mkdir(exist_ok=False);(root/'settings').mkdir();candidates=[]
    def build(name,qtag,etag,policy,model_root,calibration,environment,environment_key,pack):
        settings=root/'settings'/(name+'.json');write(settings,policy);out=root/name
        command=[sys.executable,'-B',str(Path(__file__).with_name('build_support_bundle.py')),'--models',str(model_root),'--environment-fit',str(environment),
            '--cal-pack',str(calibration),'--schema',str(s/'schema.json'),'--policy',str(settings),'--quality-tag',qtag,'--error-tag',etag,
            '--environment-key',environment_key,'--interval-method','road_envelope','--purpose','frozen_confirmation_candidate','--output',str(out)]
        subprocess.run(command,cwd=repo,check=True)
        spec=read(out/'bundle_manifest.json');candidates.append(dict(name=name,bundle=str(out),pack=str(pack),perception_identity=spec['perception_identity'],bundle_manifest_sha256=sha256_file(out/'bundle_manifest.json')))
    def qhead(family):return min([t for t in selection['quality_scores']if t.endswith('_'+family)],key=lambda t:selection['quality_scores'][t]['mae'])
    def error_policy(family):
        values=[r for r in selection['candidates']if r['error_tag'].endswith('_'+family)and r['policy']['mode']=='supported_utility'and r['policy']['beta']==0 and r['policy']['information_fraction']==0]
        return max(values,key=lambda r:(r['metrics']['utility_road_macro'],r['metrics']['near_error_road_macro']or 0.))
    key=read(Path(ready['selection'])/'environment_key.json')['key'];chosen=selection['selected']
    build('primary',selection['selected_quality_tag'],chosen['error_tag'],chosen['policy'],models,cal,envfit,key,selected_pack)
    for family in ['RF','GP']:
        error=error_policy(family);build(family+'_only',qhead(family),error['error_tag'],error['policy'],models,cal,envfit,key,selected_pack)
        if family=='GP':
            info=dict(error['policy'],beta=0.,information_fraction=.1,information_mode='conditional')
            build('GP_information_control',qhead('GP'),error['error_tag'],info,models,cal,envfit,key,selected_pack)
    qcontrol=dict(mode='quality_probability',grouping='global',beta=0.,balanced_floor=0.,audit_fraction=.1,information_fraction=0.,
                  information_mode='pointwise',utility_weights=[1.,.25,.5],reference=dict(absolute_threshold=.7,groups={}))
    build('global_q07_control',selection['selected_quality_tag'],chosen['error_tag'],qcontrol,models,cal,envfit,key,selected_pack)
    parent_selection=s/'parent_development_selection_refined';parent=read(parent_selection/'selection.json');pk=read(parent_selection/'environment_key.json')['key']
    build('parent_r1',parent['selected_quality_tag'],parent['selected']['error_tag'],parent['selected']['policy'],s/'parent_models',s/'combined_calibration_parent.npz',s/'environment_fit_refined',pk,parent_pack)
    plan=read(s/'confirmation/plan_frozen.json');choice=read(s/'protocol/perception_development_choice.json');guard=a.quality_guard or s/'protocol/quality_handoff_guard_frozen.json'
    unchanged=identity==parent_identity
    document=dict(version='scene-cycle-confirmation-freeze-1',frozen_utc=datetime.now(timezone.utc).isoformat(),confirmation_not_collected_at_freeze=True,
        study_design_sha256=sha256_file(s/'protocol/study_design_frozen.json'),confirmation_plan_sha256=sha256_file(s/'confirmation/plan_frozen.json'),
        source_git=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),source_sha256=sha256_file(Path(__file__)),
        primary_review_name='primary',parent_review_name='parent_r1',review_candidates=candidates,
        perception_models=dict(v6=dict(perception_identity=base_identity,checkpoint=str(a.baseline_checkpoint)),r1=dict(perception_identity=parent_identity,run=str(a.parent_run)),selected=dict(perception_identity=identity,run=ready['perception_run'])),
        marked_confirmation_sites=[r['route_id']for r in plan['routes']if r['expected_support_stratum']=='marked'],
        preconfirmation_perception_checks=dict(cpu_development_passes=unchanged or choice['selected']['passes'],gpu_original_retention_passes=True,gpu_report_sha256=sha256_file(a.gpu_check)),
        development_choice_sha256=sha256_file(s/'protocol/perception_development_choice.json'),models_ready_sha256=sha256_file(a.models_ready),
        preconfirmation_fallback=ready.get('preconfirmation_fallback'),
        quality_handoff_gate=read(guard),quality_handoff_gate_sha256=sha256_file(guard),
        adoption_hierarchy=['Promote the selected neural child only if frozen independent neural gates pass; otherwise retain already promoted r1.',
            'Review policy acceptance is separate. On retained r1 use only its already frozen parent_r1 policy if its own review/quality gates pass; never choose a better control after the test.',
            'A neural-only promotion does not enable automatic training admission or live ROS.'],
        scope='All candidates fixed before new confirmation collection. Family and uncertainty controls are diagnostics, not a post-test model-selection pool.')
    write(a.output,document);print('CONFIRMATION FROZEN',a.output,len(candidates),identity['checkpoint_sha256'],flush=True)


if __name__=='__main__':main()
