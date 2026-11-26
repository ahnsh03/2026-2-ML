"""Close experiment lineage, including original logs and all paired count files."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import torch
from common.bridge import load_target_model
from quality_estimation.perception_identity import load_adaptation
from quality_estimation.blend_perception import interpolate_parameters
from quality_estimation.evaluate_cached_perception import summarize
from prepare import dump


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args();study=args.study
    base=load_target_model(args.checkpoint,torch.device('cpu')).model
    initial=base.state_dict();parameter_names={n for n,_ in base.named_parameters()}
    queue=[]
    for name in ['training_queue_frozen.json','control_queue_v2_frozen.json']:
        queue+=json.loads((study/'protocol'/name).read_text())['queue']
    assert len(queue)==8
    model_seals={};replay_streams={};sources={};raw_states={}
    for item in queue:
        directory=Path(item['run']);checkpoint,metadata,membership=load_adaptation(directory)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict']
        latest=torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
        assert latest['step']==24000 and all(torch.equal(state[k],latest['model_state_dict'][k]) for k in state)
        expected_new=study/'protocol'/('replay_control_cached_v2.jsonl' if item['arm']=='ReplayOnly' else item['arm']+'_cached.jsonl')
        assert sha(directory/'new_manifest.jsonl')==sha(expected_new)
        assert sha(directory/'replay_manifest.jsonl')==sha(study/'protocol/replay_cached.jsonl')
        assert sha(directory/'protocol.json')==sha(item['protocol'])
        assert len(membership)==7168 and len({r['pixel_sha256'] for r in membership})==7168
        logged=[json.loads(line) for line in (directory/'training_steps.jsonl').read_text().splitlines()]
        steps={r['step']:r for r in logged};assert set(steps)=={1,*range(20,24001,20)}
        replay={step:row['sample_ids'][8:] for step,row in steps.items()}
        if item['seed'] in replay_streams:assert replay_streams[item['seed']]==replay
        else:replay_streams[item['seed']]=replay
        raw_states[directory.name]=state
        sources[directory.name]=metadata['source_sha256']
        model_seals[directory.name]=dict(kind='gradient_training',checkpoint_sha256=sha(checkpoint),
            run_sha256=sha(directory/'run.json'),logged_steps=len(logged),unique_logged_steps=len(steps),
            final_checkpoint_matches_optimizer_snapshot=True,
            files={str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()})
    interpolation=json.loads((study/'protocol/interpolation_queue_frozen.json').read_text())['queue']
    assert len(interpolation)==24
    for item in interpolation:
        directory=Path(item['run']);checkpoint,metadata,_=load_adaptation(directory)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict']
        expected=interpolate_parameters(initial,raw_states[Path(item['parent_run']).name],parameter_names,item['alpha'])
        assert all(torch.equal(state[k],expected[k]) for k in state)
        assert metadata['gradient_updates_performed_here']==0
        model_seals[directory.name]=dict(kind='parameter_interpolation',alpha=item['alpha'],
            checkpoint_sha256=sha(checkpoint),run_sha256=sha(directory/'run.json'),formula_tensor_exact=True,
            files={str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()})
    evaluations={}
    for marker in sorted((study/'evaluations').glob('*/*/_SUCCESS')):
        directory=marker.parent;summary=json.loads((directory/'summary.json').read_text())
        assert marker.read_text().strip()==sha(directory/'summary.json')
        rows=[json.loads(line) for line in (directory/'per_frame.jsonl').read_text().splitlines()]
        assert summary['rows']==len(rows) and summary['metrics']==summarize(rows)
        manifest=study/'protocol'/('development_retention_eval.jsonl' if directory.name=='development_retention' else 'quality_heldout_diagnostic_cached.jsonl')
        expected=[json.loads(line) for line in manifest.read_text().splitlines()]
        assert [r['sample_id'] for r in rows]==[r['sample_id'] for r in expected]
        assert summary['manifest_sha256']==sha(manifest)
        evaluations[str(directory.relative_to(study))]=dict(summary_sha256=sha(directory/'summary.json'),per_frame_sha256=sha(directory/'per_frame.jsonl'),rows=len(rows))
    private=dict(created_utc=datetime.now(timezone.utc).isoformat(),models=model_seals,evaluations=evaluations,source_variants=sources,
                 shared_logged_replay_order_verified=True,source_sha256=sha(__file__))
    dump(study/'reports/complete_study_seal_private.json',private)
    summary=dict(completed_training_runs=len(queue),parameter_derivations=len(interpolation),
                 logged_replay_order_same_within_each_seed=True,all_training_inputs_match_frozen_manifests=True,
                 final_training_checkpoints_match_optimizer_snapshots=True,interpolations_tensor_exact=True,
                 sealed_per_frame_evaluations=len(evaluations),original_first_run_training_log_sealed=True,
                 private_inventory_sha256=sha(study/'reports/complete_study_seal_private.json'),source_sha256=sha(__file__))
    dump(study/'reports/complete_study_verification_summary.json',summary)
    print('STUDY SEALED',len(queue),len(interpolation),len(evaluations),flush=True)


if __name__=='__main__':main()
