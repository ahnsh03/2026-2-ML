"""Resume a frozen low-speed cohort, with a parked service epoch per drive."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def windows(path):
    return 'E:/' + str(path.resolve().relative_to('/mnt/e'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--cohort',choices=['scout','development','calibration','confirmation','adaptation'],required=True)
    parser.add_argument('--reset-script',type=Path,required=True)
    parser.add_argument('--collector',default='ml-scene-collector')
    parser.add_argument('--hard-stop-utc',required=True)
    parser.add_argument('--capture-interval',type=float,default=1.2)
    args=parser.parse_args()
    study=args.study.resolve();root=study/args.cohort;root.mkdir(exist_ok=True)
    (root/'runs').mkdir(exist_ok=True)
    plan=read(args.plan);plan_digest=hashlib.sha256(args.plan.read_bytes()).hexdigest()
    needed={(r['route_id'],w,h)for r in plan['routes']for w,h in r['conditions']}
    if not needed:raise ValueError('empty collection cohort')
    deadline=datetime.fromisoformat(args.hard_stop_utc)
    epoch=0;failures=0;consecutive_failures=0
    while datetime.now(timezone.utc)<deadline:
        good=[read(p.parent/'summary.json')for p in (root/'runs').glob('*/_SUCCESS')]
        good=[r for r in good if r['plan_sha256']==plan_digest]
        done={(r['route_id'],r['weather'],r['hour'])for r in good}
        prefixes=sorted({r['run_id'].rsplit('_'+r['route_id']+'_w',1)[0]for r in good})
        if needed<=done:
            result=dict(status='complete',conditions=len(done),frames=sum(r['frames']for r in good),
                        role=args.cohort,finished_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=plan_digest)
            (root/'COLLECTION_COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
            print('COHORT COMPLETE',args.cohort,result,flush=True);return
        prefix=f'{args.cohort}28e{epoch:03d}'
        if (root/(prefix+'_before.json')).exists()or(study/'logs'/f'collect_{prefix}.log').exists():
            epoch+=1;continue
        with (study/'logs'/f'reset_{prefix}.log').open('x')as log:
            result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',windows(args.reset_script),
                '-RelayRoot',windows(study/'udp_relay'),'-Receipt',windows(root/(prefix+'_rpc_epoch.json'))],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('parked RPC reset failed; inspect before further driving')
        command=['docker','exec',args.collector,'python','/workspace/scripts/experiments/roi9_drive_20261126/collect.py',
            '--plan','/experiment/'+str(args.plan.resolve().relative_to(study)),
            '--output','/experiment/'+args.cohort,'--simulator-file-dir',windows(root/'captures'),
            '--prefix',prefix,'--udp-relay','/experiment/udp_relay','--steering-range-deg','40','--target-kph','4',
            '--native-velocity-divisor','3.6','--capture-interval',str(args.capture_interval),
            '--hard-stop-utc',args.hard_stop_utc,'--max-new-runs','1','--reuse-prefix',*prefixes]
        with (study/'logs'/f'collect_{prefix}.log').open('x')as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        completion=root/(prefix+'_completion.json')
        if not completion.exists()or not read(completion)['restoration'].get('verified'):
            raise RuntimeError('missing verified restoration; recover before further driving')
        if result.returncode:
            failures+=1
            consecutive_failures+=1
            if consecutive_failures>=3:raise RuntimeError('three consecutive failed service epochs; retry bound exhausted')
        else:
            consecutive_failures=0
        print('EPOCH',prefix,'completed',len(done),'/',len(needed),'failures',failures,'consecutive',consecutive_failures,flush=True)
        epoch+=1
    raise TimeoutError('cohort hard deadline reached')


if __name__=='__main__':
    main()
