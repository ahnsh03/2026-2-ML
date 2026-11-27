"""Run fixed-v6, full-original-replay arms; adaptive confirmations use development only."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,subprocess,time,hashlib

def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args();S=a.study
    replay=S/'protocol/full_original_cached.jsonl'
    while not (S/'reports/full_original_cache_audit.json').exists():time.sleep(10)
    assert len(replay.read_text().splitlines())==26001
    # Every first-stage arm shares seed/training settings. The separate frozen
    # design states the rule for a second optimizer seed and equal-budget tests.
    while True:
        plan=json.loads((S/'protocol/training_plan.json').read_text())
        pending=[r for r in plan['queue'] if not (Path(r['run'])/'_SUCCESS').exists()]
        if not pending:
            if plan.get('queue_closed'):break
            time.sleep(10);continue
        item=pending[0];manifest=Path(item['manifest'])
        if not manifest.exists():time.sleep(10);continue
        protocol=S/'protocol'/('training_'+str(item['seed'])+'.json')
        settings=dict(schema_version='long-adaptation-1',selection_arms=plan['allowed_arms'],seed=item['seed'],steps=24000,new_per_batch=2,replay_per_batch=14,batch_size=16,
                      learning_rate=1e-5,minimum_lr_factor=.1,warmup_steps=200,weight_decay=1e-4,gradient_clip_norm=10.,precision='BF16',loss_precision='FP32 reductions',workers=4,checkpoint_every=500,
                      replay_distillation_weight=.25,distillation_temperature=2.,augmentation='none',normalization='BN running moments frozen, affine trainable',
                      original_replay_rows=26001,new_rows=3072,expected_new_pool_passes=15.625,expected_original_pool_passes=336000/26001,
                      interpretation='Original frequencies preserved by full-pool permutations. Explicit new2/original14 batch ratio; all runs start same v6, no heldout selection.')
        if not protocol.exists():write(protocol,settings)
        elif json.loads(protocol.read_text())!=settings:raise ValueError('training protocol changed')
        write(S/'WORK_IN_PROGRESS.json',dict(status='training',current=item,updated_utc=datetime.now(timezone.utc).isoformat(),completed=[r['name'] for r in plan['queue'] if (Path(r['run'])/'_SUCCESS').exists()]))
        command=['docker','run','--name','ml-environment-training','--rm','--gpus','all','--shm-size','1g',
                 '-v',str(a.repo)+':/repo:ro','-v','/mnt/e:/mnt/e','-v',str(a.checkpoints)+':/checkpoints:ro','-w','/repo',
                 '-e','PYTHONPATH=/repo/src:/repo/ros/src/data_collection/src','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',
                 'ml-perception:rtx50-cu128','python','-m','quality_estimation.retrain_streaming','--new-manifest',str(manifest),'--replay-manifest',str(replay),
                 '--protocol',str(protocol),'--cache',str(a.cache),'--arm',item['arm'],'--checkpoint','/checkpoints/twinlite_medium_morai_v6_road_marking_init_b_b12_e18_20260908/best_mean.pt','--output-dir',item['run']]
        if (Path(item['run'])/'latest.pt').exists():command.append('--resume')
        with (S/'logs'/('train_'+item['name']+'.log')).open('a') as f:
            print('START',item,flush=True);subprocess.run(command,cwd=a.repo,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',item['name'],flush=True)
    write(S/'WORK_IN_PROGRESS.json',dict(status='training_complete',updated_utc=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__':main()
