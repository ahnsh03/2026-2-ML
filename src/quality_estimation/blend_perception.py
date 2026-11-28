"""Derive a fixed intermediate update from compatible f0/finetuned parameters.

This performs no gradient updates and runs one network at inference. It is an
additional development experiment inspired by weight interpolation, not a
reproduction of CLIP/WiSE-FT or a guarantee of retention.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
from common.bridge import DEFAULT_CHECKPOINT_SHA256,load_target_model,sha256_file
from .perception_identity import load_adaptation
from .perception_data import POLICY_SHA256
from .artifacts import write_json

def interpolate_parameters(initial,adapted,parameter_names,alpha):
    if not np.isfinite(alpha) or not 0<=alpha<=1 or set(initial)!=set(adapted):raise ValueError('incompatible parameter interpolation')
    if not set(parameter_names).issubset(initial):raise ValueError('unknown interpolation parameter')
    result={}
    for name,left in initial.items():
        right=adapted[name]
        if left.shape!=right.shape or left.dtype!=right.dtype:raise ValueError('state shape/dtype differs: '+name)
        if name in parameter_names:
            if not left.is_floating_point() or not torch.isfinite(left).all() or not torch.isfinite(right).all():raise ValueError('invalid floating parameter')
            result[name]=left.clone() if alpha==0 else right.clone() if alpha==1 else (1-alpha)*left+alpha*right
        else:
            if not torch.equal(left,right):raise ValueError('non-parameter buffers changed; cannot interpolate: '+name)
            result[name]=left.clone()
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['initial-checkpoint','parent-run','extension-protocol','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--alpha',type=float,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('interpolation output exists')
    protocol=json.loads(a.extension_protocol.read_text())
    if a.alpha not in protocol['alphas'] or not 0<a.alpha<1:raise ValueError('interpolation coefficient not registered')
    if sha256_file(a.initial_checkpoint)!=DEFAULT_CHECKPOINT_SHA256:raise ValueError('initial v6 EMA differs')
    parent_path,parent,_=load_adaptation(a.parent_run);state=torch.load(parent_path,map_location='cpu',weights_only=False)
    base=load_target_model(a.initial_checkpoint,torch.device('cpu'));initial=base.model.state_dict();adapted=state['model_state_dict'];names={n for n,_ in base.model.named_parameters()}
    parameters=interpolate_parameters(initial,adapted,names,a.alpha)
    out=a.output;out.mkdir(parents=True)
    copy_protocol=dict(protocol,selected_alpha=a.alpha,parent_run_sha256=sha256_file(a.parent_run/'run.json'),parent_checkpoint_sha256=sha256_file(parent_path))
    write_json(out/'protocol.json',copy_protocol)
    shutil.copy2(a.parent_run/'training_membership.json',out/'training_membership.json');shutil.copy2(a.parent_run/'run.json',out/'parent_run.json')
    (out/'source').mkdir();shutil.copy2(Path(__file__),out/'source/blend_perception.py')
    result=dict(model_state_dict=parameters,task_spec=base.task_spec.as_dict(),upstream_commit=base.upstream_commit,epoch=None,
                adaptation_steps=parent['steps'],initial_checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,source_weights='f0_ema_plus_parent_model',
                adaptation_protocol_sha256=sha256_file(out/'protocol.json'),interpolation_alpha=a.alpha,parent_checkpoint_sha256=sha256_file(parent_path))
    torch.save(result,out/'checkpoint.pt')
    metadata=dict(schema_version='ml-perception-adaptation-1',status='complete',initial_checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,target_policy_sha256=POLICY_SHA256,
        derivation='fixed_parameter_interpolation',alpha=a.alpha,parent_run_sha256=sha256_file(a.parent_run/'run.json'),parent_checkpoint_sha256=sha256_file(parent_path),
        seed=parent['seed'],steps=parent['steps'],gradient_updates_performed_here=0,additional_training_seconds=0,new_rows=parent['new_rows'],replay_rows=parent['replay_rows'],
        train_parent_groups=parent['train_parent_groups'],checkpoint_sha256=sha256_file(out/'checkpoint.pt'),source_sha256=sha256_file(Path(__file__)),
        quality_regressor_recalibrated_for_f1=False,scope='One static derived network. All coefficients evaluated on development/retention only; heldout labels cannot select alpha.')
    # Interpolation is a derivation of this training generation, not another
    # training update. Retain its cumulative ancestry for the next update.
    metadata['generation']=int(parent.get('generation',1))
    for name in ['parent_perception_identity','training_start_checkpoint_sha256','cumulative_membership','update_mode']:
        if name in parent:metadata[name]=parent[name]
    artifacts=['checkpoint.pt','training_membership.json','protocol.json','parent_run.json','source/blend_perception.py']
    if (a.parent_run/'parent_lineage.json').exists():
        shutil.copy2(a.parent_run/'parent_lineage.json',out/'parent_lineage.json');artifacts.append('parent_lineage.json')
    metadata['artifact_sha256']={name:sha256_file(out/name) for name in artifacts}
    write_json(out/'run.json',metadata);(out/'_SUCCESS').write_text(sha256_file(out/'run.json')+'\n');print('DERIVED',a.alpha,metadata['checkpoint_sha256'],flush=True)
if __name__=='__main__':main()
