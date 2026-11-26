"""Measured vehicle motion and independent heartbeat-loss stop figures."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def read(path):return json.loads(path.read_text())
def save(fig,root,name):fig.savefig(root/(name+'.png'),dpi=160);fig.savefig(root/(name+'.svg'));plt.close(fig)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--watchdog',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);n=a.study
 plan=read(n/'plans/routes_geometry_frozen.json');routes={r['route_id']:r for r in plan['routes']};color={'development':'#2171b5','calibration':'#e39d35','test':'#45a278'}
 representatives={next(r['route_id'] for r in plan['routes'] if r['role']==role) for role in color}
 fig=plt.figure(figsize=(16,9));gs=GridSpec(2,2,figure=fig,width_ratios=[1.15,1]);mapax=fig.add_subplot(gs[:,0]);speedax=fig.add_subplot(gs[0,1]);cteax=fig.add_subplot(gs[1,1]);seen=set()
 canonical={r['run_id'] for role in color for r in read(n/('registry_'+role+'_final/audit.json'))['runs']}
 for path in sorted((n/'runs').glob('*/summary.json')):
  if path.parent.name not in canonical:continue
  s=read(path)
  if s['weather']!=0 or s['hour']!=13 or s['role'] not in color:continue
  if s['route_id'] not in routes:continue
  control=[json.loads(line) for line in (path.parent/'control.jsonl').read_text().splitlines()];xy=np.array([[r['state']['transform']['location'][k] for k in 'xy'] for r in control]);points=np.asarray(routes[s['route_id']]['points']);role=s['role'];c=color[role]
  mapax.plot(points[:,0],points[:,1],'--',color='black',lw=2,alpha=.6);mapax.plot(xy[:,0],xy[:,1],color=c,lw=1.3,label=role if role not in seen else None);seen.add(role);mapax.text(xy[0,0]+2,xy[0,1]+2,s['route_id'].split('_')[-1],fontsize=8)
  if s['route_id'] in representatives:
   station=[r['station_m'] for r in control];speedax.plot(station,[r['speed_mps']*3.6 for r in control],color=c,label=s['route_id']);cteax.plot(station,[r['cte_m']*100 for r in control],color=c,label=s['route_id'])
 mapax.set_aspect('equal');mapax.set(xlabel='Local map X (m)',ylabel='Local map Y (m)',title='Nine separated new segments on the same map');mapax.legend();speedax.set(xlabel='Measured route progress (m)',ylabel='Vehicle speed (km/h)',title='Actual low-speed control (one segment per role)');speedax.legend();cteax.set(xlabel='Measured route progress (m)',ylabel='Cross-track error (cm)',title='Tracking error from independent vehicle telemetry');cteax.legend()
 fig.suptitle('135 completed drives: new geometry with controlled weather/time',fontsize=20);fig.text(.03,.025,'Dashed black: planned route. Colored: measured moving trajectory at Sunny13. Other conditions are retained in raw logs.\nTarget4 km/h; maxima and full-run integrity are reported separately. Telemetry time is not camera exposure time.',fontsize=11);fig.subplots_adjust(bottom=.16,top=.87,wspace=.3,hspace=.4);save(fig,a.output,'06_actual_motion')
 watchdog=a.watchdog or (n/'watchdog_test_v4' if (n/'watchdog_test_v4/result.json').exists() else n/'watchdog_test')
 record=watchdog/'telemetry.json'
 if record.exists() and read(watchdog/'result.json').get('status')=='passed':
  rows=read(record);pre=[r for r in rows if r['stage']=='heartbeat'];lost=[r for r in rows if r['stage']=='heartbeat_lost'];origin=max(r['elapsed'] for r in pre);times=[r['elapsed']-origin if r['stage']=='heartbeat' else r['elapsed'] for r in rows]
  fig,ax=plt.subplots(figsize=(16,9));ax.plot(times,[r['speed_mps']*3.6 for r in rows],color='#2171b5',label='Measured speed');ax.axvline(0,color='black',ls='--',label='Command heartbeat stopped');ax.axvline(.6,color='#d14c4c',ls=':',label='Configured watchdog0.6s');ax.set(xlabel='Time relative to heartbeat loss (s)',ylabel='Vehicle speed (km/h)',title='Independent UDP watchdog: one low-speed fault injection');ax.legend();result=read(watchdog/'result.json');fig.text(.05,.04,'Observed status: '+result['status']+'. One straight low-speed probe; this is not a general braking/safety certification.',fontsize=12);fig.subplots_adjust(bottom=.15);save(fig,a.output,'07_watchdog_stop')
 print('motion figures complete',flush=True)
if __name__=='__main__':main()
