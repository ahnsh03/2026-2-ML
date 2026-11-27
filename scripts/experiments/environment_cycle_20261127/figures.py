"""Standalone aggregate figures; fresh evidence is read only after frozen assessment."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_data import read,dump,sha,load_sources,memberships,labels,macro

COLORS=['#527ba5','#e5a14b','#4c9b89','#925c9c','#bf6056','#64707d']
def save(fig,out,name):
 fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();S=a.study;out=S/'figures';out.mkdir(exist_ok=True);plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white'})
 replay=read(S/'reports/original_replay_audit.json');x=np.arange(12);fig,ax=plt.subplots(figsize=(12,5));ax.bar(x-.18,[r['old_replay_fraction']*100 for r in replay['runs']],.36,label='Previous4,096 subset',color=COLORS[1]);ax.bar(x+.18,[r['original_fraction']*100 for r in replay['runs']],.36,label='Complete26,001 replay',color=COLORS[0]);ax.set(xticks=x,xticklabels=[f'Run{i+1}' for i in range(12)],ylabel='Share of original replay (%)',title='Full replay preserves the original acquisition mixture');ax.legend();fig.tight_layout();save(fig,out,'01_original_replay_composition')
 env=read(S/'environment/environment_selection_frozen.json');raw=next(r for r in env['clusters'] if r['method']=='official_e_means' and r['representation']=='rgb27' and r['k']==16 and r['seed']==271127);learned=env['selected_cluster'];classifier=env['selected_classifier'];names=['RGB27 + e-means','RGB400 classifier','RGB probabilities + e-means'];wx=[raw['dev_weather'],classifier['dev_weather'],learned['dev_weather']];lx=[raw['dev_lighting'],classifier['dev_lighting'],learned['dev_lighting']];fig,ax=plt.subplots(figsize=(12,5));x=np.arange(3);ax.bar(x-.18,wx,.36,label='Weather balanced accuracy',color=COLORS[0]);ax.bar(x+.18,lx,.36,label='Lighting balanced accuracy',color=COLORS[2]);ax.set(xticks=x,xticklabels=names,ylim=(0,1.09),title='Environment representation on three development roads',ylabel='Balanced accuracy');ax.legend(loc='lower right');fig.text(.08,.01,'Supervised RGB representation is not label-free discovery. Dev contains5 of6 weather classes.',fontsize=10,color='#555');fig.tight_layout(rect=(0,.04,1,1));save(fig,out,'02_environment_development')
 banks=read(S/'reports/admission_summary.json');arms=['RandomFull','TagBalancedFull','RFGlobalFull','GPGlobalFull'];fig,axes=plt.subplots(1,2,figsize=(13,5));weather_names=['Sunny','Cloudy','Foggy','Storm','Rainy','Snowy'];bottom=np.zeros(4)
 for w in range(6):
  values=np.array([banks[arm]['weather_counts'].get(str(w),0) for arm in arms])/3072*100;axes[0].bar(np.arange(4),values,bottom=bottom,label=weather_names[w],color=COLORS[w]);bottom+=values
 bottom=np.zeros(4)
 for l,label,color in [(0,'Day',COLORS[1]),(1,'Transition',COLORS[2]),(2,'Night',COLORS[0])]:
  values=np.array([banks[arm]['lighting_counts'].get(str(l),0) for arm in arms])/3072*100;axes[1].bar(np.arange(4),values,bottom=bottom,label=label,color=color);bottom+=values
 for ax,title in zip(axes,['Weather mixture','Lighting mixture']):ax.set(xticks=np.arange(4),xticklabels=['Random','Balanced','RF risk','GP risk'],ylabel='New admission bank (%)',title=title);ax.legend(fontsize=9,loc='upper right')
 fig.suptitle('Same3,072 new slots; complete26,001 originals in every arm',fontsize=15);fig.tight_layout();save(fig,out,'03_admission_environment_balance')
 data,_,rows=load_sources(a.parent);m=memberships(S,data['ids']);ix=m['development'];ix=ix[np.array([rows[i]['source_key'].startswith('drive_development_') for i in ix])];quality=read(S/'f0_quality/quality_selection_frozen.json');q,_,_=labels(data);fig,axes=plt.subplots(1,2,figsize=(12,5))
 for ax,kind in zip(axes,['GP','RF']):
  r=quality['selected'][kind];z=np.load(S/r['prediction']);ax.scatter(q[ix],z['mean'][ix],c=data['hour'][ix],cmap='viridis',s=13,alpha=.6);ax.plot([0,1],[0,1],'--',color='#777',lw=1);ax.set(xlim=(0,1),ylim=(0,1),xlabel='GT whole-frame lane IoU',ylabel='Predicted IoU',title=f"{kind}: {r['view']} / road-macro MAE={r['dev_primary_mae']:.4f}")
 from matplotlib.patches import Patch
 fig.legend(handles=[Patch(color=plt.cm.viridis((h-8)/12),label=f'{h:02d}:00 simulator time') for h in [8,13,20]],loc='lower center',ncol=3,frameon=False)
 fig.suptitle('GT-free quality regression; identical quality-fit membership');fig.tight_layout(rect=(0,.08,1,1));save(fig,out,'04_f0_quality_regression')
 if (S/'reports/neural_development_partial.json').exists():
  d=read(S/'reports/neural_development_partial.json');records=d['records']
  if records:
   fig,ax=plt.subplots(figsize=(12,5))
   for j,arm in enumerate(arms):
    rs=[r for r in records if r['arm']==arm and r['alpha']==1.]
    if rs:ax.scatter([j]*len(rs),[r['gain'] for r in rs],label=arm,color=COLORS[j],s=70)
   ax.axhline(0,color='#888',lw=1);ax.set(xticks=np.arange(4),xticklabels=['Random','Balanced','RF risk','GP risk'],ylabel='Development lane-IoU gain over v6',title='Equal24,000-update endpoints with full original replay');fig.tight_layout();save(fig,out,'05_neural_development_raw')
 if (S/'reports/fresh_confirmation_assessment.json').exists():
  d=read(S/'reports/fresh_confirmation_assessment.json');fig,axes=plt.subplots(1,2,figsize=(12,5));roads=list(d['neural']['f0']['roads']);x=np.arange(len(roads))
  for dx,name,color in [(-.18,'f0',COLORS[0]),(.18,'f1',COLORS[2])]:axes[0].bar(x+dx,[d['neural'][name]['roads'][r] for r in roads],.36,label=name,color=color)
  axes[0].set(xticks=x,xticklabels=[f'Road{i+1}' for i in range(len(roads))],ylim=(0,1),ylabel='Mean supported frame IoU',title='Fresh three-road confirmation');axes[0].legend()
  names=['f0/g0','f1/g1','f1/stale g0'];values=[d['quality']['f0']['macro_road_mae'],d['quality']['f1']['macro_road_mae'],d['stale_g0_on_f1_macro_road_mae']];axes[1].bar(np.arange(3),values,color=[COLORS[0],COLORS[2],COLORS[1]]);axes[1].set(xticks=np.arange(3),xticklabels=names,ylabel='Road-macro IoU MAE',title='Re-estimate quality after perception changes');fig.tight_layout();save(fig,out,'06_fresh_neural_and_quality')
  fig,ax=plt.subplots(figsize=(12,5));e=d['environment'];names=['RGB weather','Cluster weather','RGB lighting','Cluster lighting'];values=[e['direct_weather_balanced_accuracy'],e['cluster_weather_balanced_accuracy'],e['direct_lighting_balanced_accuracy'],e['cluster_lighting_balanced_accuracy']];ax.bar(np.arange(4),values,color=[COLORS[0],COLORS[2],COLORS[0],COLORS[2]]);ax.set(xticks=np.arange(4),xticklabels=names,ylim=(0,1.05),ylabel='Balanced accuracy',title='Fresh confirmation: six weather presets and three lighting bins');fig.tight_layout();save(fig,out,'07_fresh_environment')
 if (S/'reports/fresh_confirmation_assessment.json').exists():
  d=read(S/'reports/fresh_confirmation_assessment.json');names=['RandomFull','TagBalancedFull','RFGlobalFull','GPGlobalFull','ReplayFull'];labels_=['Random','Balanced random','RF risk','GP risk','Original only'];fig,ax=plt.subplots(figsize=(12,5))
  for j,arm in enumerate(names):
   values=[r['metrics']['macro'] for r in d['matched_neural_methods'] if r['arm']==arm];ax.bar(j,np.mean(values),color=COLORS[j],alpha=.75);ax.scatter([j]*len(values),values,color='#20252b',s=35,zorder=3)
  ax.axhline(d['neural']['f0']['macro'],color='#666',linestyle='--',label='Original v6');ax.set(xticks=np.arange(5),xticklabels=labels_,ylim=(0,1),ylabel='Supported road-macro lane IoU',title=f"Fresh comparison at the same alpha={d['matched_alpha']:.2f}");ax.legend();fig.text(.08,.015,'Points are optimizer seeds of one fixed admission bank; balanced/original-only each have one seed.',fontsize=10);fig.tight_layout(rect=(0,.04,1,1));save(fig,out,'09_fresh_selection_comparison')
 dump(out/'inventory.json',{p.name:sha(p) for p in out.iterdir() if p.suffix in ['.png','.svg']});print('FIGURES',len(list(out.glob('*.png'))),flush=True)
if __name__=='__main__':main()
