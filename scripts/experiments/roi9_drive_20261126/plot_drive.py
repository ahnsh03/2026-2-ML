"""Show planned paths and actual recorded vehicle motion, without model fitting."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extension',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();e=a.extension;a.output.mkdir(parents=True,exist_ok=False)
    plan=read(e/'plans/frozen.json');runs=[(p.parent,read(p)) for p in (e/'runs').glob('*/summary.json')]
    fig,axes=plt.subplots(2,3,figsize=(16,9));provenance=[]
    for j,route in enumerate(plan['routes']):
        origin=np.asarray(route['points'][0])[:2];planned=np.asarray(route['points'])[:,:2]-origin
        ax=axes[0,j];ax.plot(planned[:,0],planned[:,1],'k--',lw=2,label='Planned32m')
        for w,h,color,label in [(0,13,'#174d78','Sunny13'),(4,13,'#4c9b91','Rain13'),(5,20,'#d68c35','Snow20')]:
            root,summary=next((root,s) for root,s in runs if s['route_id']==route['route_id'] and s['weather']==w and s['hour']==h)
            rows=[json.loads(line) for line in (root/'control.jsonl').read_text().splitlines()]
            xy=np.array([[r['state']['transform']['location'][axis] for axis in 'xy'] for r in rows])-origin
            ax.plot(xy[:,0],xy[:,1],color=color,lw=1.5,alpha=.85,label=label)
            axes[1,j].plot([r['elapsed'] for r in rows],[r['speed_mps']*3.6 for r in rows],color=color,lw=1,label=label)
            provenance.append(dict(route_id=route['route_id'],weather=w,hour=h,run_id=root.name,control_sha256=sha(root/'control.jsonl'),maximum_cte_m=summary['maximum_cte_m'],maximum_speed_kph=summary['maximum_speed_mps']*3.6))
        ax.set_aspect('equal',adjustable='datalim');ax.set(xlabel='Ego X relative to planned start (m)',ylabel='Ego Y relative to planned start (m)',title=f'Fresh segment {chr(65+j)}');ax.legend(fontsize=9)
        ax=axes[1,j];ax.axhline(4,color='#777',ls='--',label='Target4km/h');ax.set(xlabel='Elapsed driving time (s)',ylabel='Measured speed (km/h)',ylim=(0,6.5));ax.legend(fontsize=9)
    fig.suptitle('Actual low-speed26R1 driving on separated new segments',fontsize=20)
    fig.text(.03,.025,'Recorded vehicle telemetry; native speed conversion verified using displacement. Three shown conditions out of15 per segment.\nStops before the end of each planned segment. No direct ROS, camera exposure synchronization or full-map route-completion claim.',fontsize=11)
    fig.tight_layout(rect=(0,.08,1,.94))
    for suffix in ['png','svg']:fig.savefig(a.output/('actual_motion.'+suffix),dpi=160)
    plt.close(fig)
    (a.output/'provenance.json').write_text(json.dumps(dict(plan_sha256=sha(e/'plans/frozen.json'),shown_runs=provenance),indent=2)+'\n')


if __name__=='__main__':main()
