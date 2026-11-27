"""Collect quarantined confirmation drives using explicit parked service epochs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,subprocess,time

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--reset-script',type=Path,required=True);p.add_argument('--hard-stop-utc',required=True);p.add_argument('--plan-name',default='frozen.json');p.add_argument('--prefix-base',default='env28e');p.add_argument('--completion-name',default='COLLECTION_COMPLETE.json');a=p.parse_args();S=a.study;C=S/'confirmation';plan_file=C/'plans'/a.plan_name;plan=json.loads(plan_file.read_text());needed={(r['route_id'],w,h) for r in plan['routes'] for w,h in r['conditions']};epoch=0;failures=0
 while datetime.now(timezone.utc)<datetime.fromisoformat(a.hard_stop_utc):
  good=[]
  for success in (C/'runs').glob('*/_SUCCESS'):
   summary=json.loads((success.parent/'summary.json').read_text())
   if summary['plan_sha256']==__import__('hashlib').sha256(plan_file.read_bytes()).hexdigest():good.append(summary)
  done={(r['route_id'],r['weather'],r['hour']) for r in good};prefixes=sorted({r['run_id'].split('_env28_')[0] for r in good})
  if needed<=done:
   (C/a.completion_name).write_text(json.dumps(dict(status='complete',conditions=len(done),frames=sum(r['frames'] for r in good),role='fresh_confirmation_test',finished_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n');print('COLLECTION COMPLETE',len(done),flush=True);return
  prefix=f'{a.prefix_base}{epoch:03d}'
  if (C/(prefix+'_before.json')).exists() or (S/'logs'/f'collect_{prefix}.log').exists():epoch+=1;continue
  win_script='E:/'+str(a.reset_script.relative_to('/mnt/e'));receipt=f'E:/'+str((C/f'rpc_epoch_{epoch:03d}.json').relative_to('/mnt/e'))
  with (S/'logs'/f'collect_reset_{epoch:03d}.log').open('a') as log:
   result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',win_script,'-RelayRoot','E:/'+str((S/'udp_relay').relative_to('/mnt/e')),'-Receipt',receipt],stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError('parked service reset rejected; inspect window and state, no blind retry')
  command=['docker','exec','ml-environment-collector','python','/workspace/scripts/experiments/roi9_drive_20261126/collect.py','--plan','/experiment/confirmation/plans/'+a.plan_name,'--output','/experiment/confirmation','--simulator-file-dir','E:/'+str((C/'captures').relative_to('/mnt/e')),'--prefix',prefix,'--udp-relay','/experiment/udp_relay','--steering-range-deg','40','--target-kph','4','--native-velocity-divisor','3.6','--capture-interval','1.2','--hard-stop-utc',a.hard_stop_utc,'--max-new-runs','1','--reuse-prefix',*prefixes]
  with (S/'logs'/f'collect_{prefix}.log').open('x') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:
   failures+=1
   completion=C/(prefix+'_completion.json')
   if not completion.exists() or not json.loads(completion.read_text())['restoration'].get('verified'):raise RuntimeError('collection failed restoration; recover before driving')
   if failures>=3:raise RuntimeError('three collector failures, bounded retry exhausted')
  print('COLLECTED_EPOCH',epoch,'remaining_before',len(needed-done),flush=True);epoch+=1
 raise TimeoutError('confirmation collection deadline')
if __name__=='__main__':main()
