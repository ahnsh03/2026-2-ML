"""Standalone 16:9 research figures from frozen recipe/admission evidence."""
from pathlib import Path
import argparse,json,hashlib
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from quality_estimation.spatial_features import SpatialProfile

NAVY='#102f59';TEAL='#168e91';GOLD='#db9d30';GRAY='#65758a';WEATHER=['Sunny','Cloudy','Foggy','Storm','Rainy','Snowy']
def canvas(title):
 fig=plt.figure(figsize=(12,6.75),facecolor='white');fig.suptitle(title,x=.05,y=.96,ha='left',fontsize=20,color=NAVY,weight='bold');return fig
def save(fig,out,name,caption):
 fig.text(.05,.025,caption,fontsize=10,color=GRAY);fig.savefig(out/(name+'.png'),dpi=160);fig.savefig(out/(name+'.svg'));plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();S=a.study;out=a.output;out.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
 fig=canvas('A versioned, GT-free review loop');ax=fig.add_axes([.04,.15,.92,.7]);ax.axis('off')
 blocks=[('Capture & infer','Front RGB\nPerception model $f_v$'),('Estimate & group','9-region features\nGP quality + e-means context'),('Select & verify','Fixed admission budget\nOffline native Semantic GT'),('Update & rebind','New labels + original replay\n$f_{v+1}$, then regenerate $g_{v+1}$')]
 for i,(title,body) in enumerate(blocks):
  x=.01+i*.25;box=FancyBboxPatch((x,.34),.22,.42,boxstyle='round,pad=.01,rounding_size=.025',facecolor='#eff4fa' if i<2 else '#eaf6f3',edgecolor=NAVY if i<2 else TEAL,linewidth=1.8);ax.add_patch(box);ax.text(x+.11,.65,title,ha='center',fontsize=13,weight='bold',color=NAVY);ax.text(x+.11,.50,body,ha='center',va='center',fontsize=10.5,linespacing=1.7,color=NAVY)
  if i<3:ax.add_patch(FancyArrowPatch((x+.225,.55),(x+.246,.55),arrowstyle='-|>',mutation_scale=15,color=NAVY))
 ax.text(.25,.9,'No GT, weather ID or clock as an inference input',ha='center',fontsize=12,color=NAVY);ax.text(.76,.9,'GT only after review selection',ha='center',fontsize=12,color=TEAL)
 ax.annotate('Offline BO: measured development objectives, with equal-budget random search',xy=(.4,.30),xytext=(.5,.11),ha='center',fontsize=11,color=GRAY,arrowprops=dict(arrowstyle='->',color=GRAY,linestyle='--'))
 save(fig,out,'01_versioned_pipeline','Clusters guide coverage/audit. Absolute low-IoU benchmark stays 0.7; operating normality is a separate question.')
 rows=[json.loads(s) for s in (S/'protocol/all_sources_v2.jsonl').read_text().splitlines()];profile=SpatialProfile.load();sample=next(r for r in rows if r['source_key'].startswith('drive_development_') and r['weather']==0 and r['hour']==13)
 image=cv2.cvtColor(cv2.imread(str(Path('/mnt')/sample['image'])),cv2.COLOR_BGR2RGB);mask=profile.native;colors=np.array([[40,110,185],[70,155,215],[60,185,170],[50,145,135],[235,175,60],[220,145,40],[220,100,80],[190,75,95],[100,105,120]])/255.
 fig=canvas('Fixed 9-region geometry; different roles for different features');ax=fig.add_axes([.05,.16,.63,.73]);ax.imshow(image);ax.imshow(colors[mask],alpha=.24)
 for k,name in enumerate(profile.region_names):
  yy,xx=np.where(mask==k);ax.text(xx.mean(),yy.mean(),name.replace('upper_','').replace('_','\n'),ha='center',va='center',fontsize=10,color='white',bbox=dict(facecolor=NAVY,alpha=.75,edgecolor='none',pad=2))
 ax.axis('off');ax=fig.add_axes([.73,.25,.23,.53]);ax.axis('off');ax.text(0,.98,'Quality: 28 inputs',fontsize=15,color=NAVY,weight='bold',va='top');ax.text(0,.84,'Near-road appearance\nPerception output statistics\nGlobal uncertainty\nNear–far output relations',fontsize=12,color=NAVY,linespacing=1.7,va='top');ax.text(0,.38,'Environment: 27 inputs',fontsize=14,color=TEAL,weight='bold',va='top');ax.text(0,.25,'Upper-top left/right + hood\n9 photometric statistics each\nRGB only; no perception output',fontsize=11,color=TEAL,linespacing=1.7,va='top')
 save(fig,out,'02_roi_and_feature_roles','Image bands, not semantic classes or metric depth. Quality/features may use pooled regions and the whole image.')
 study=json.loads((S/'clusters/study.json').read_text());records=study['records'];fig=canvas('Compact RGB context descriptors for e-means');axes=[fig.add_axes([.08,.22,.38,.61]),fig.add_axes([.57,.22,.37,.61])]
 palette=[NAVY,TEAL,GOLD,'#a05d78']
 for color,view in zip(palette,sorted({r['view'] for r in records})):
  rs=sorted([r for r in records if r['view']==view],key=lambda r:r['k']);x=[r['k'] for r in rs]
  axes[0].plot(x,[.5*(r['mean_weather_nmi']+r['mean_hour_nmi']) for r in rs],marker='o',label=view,color=color);axes[1].plot(x,[r['seed_ari'] for r in rs],marker='o',label=view,color=color)
 for ax,title in zip(axes,['Environment information (mean NMI)','Stability across 3 seeds (ARI)']):ax.set_title(title,fontsize=13);ax.set_xticks([8,16,32]);ax.set_xlabel('Number of clusters');ax.set_ylim(0,1);ax.grid(alpha=.2)
 axes[0].legend(fontsize=9,loc='lower left');selected=study['selected'];fig.text(.56,.12,f"Selected: {selected['view']}, K={selected['k']}",color=TEAL,fontsize=13,weight='bold')
 save(fig,out,'03_environment_recipe','NMI/ARI are clustering diagnostics, not weather-classification accuracy. Selection used training-candidate RGB/metadata only.')
 admissions=json.loads((S/'reports/admission_summary.json').read_text());fig=canvas('Same 3,072-image budget; different environment coverage');ax=fig.add_axes([.08,.23,.86,.61]);x=np.arange(selected['k']);width=.24
 for j,(name,color) in enumerate(zip(['Random','QualityRank','ClusterBO'],[GRAY,NAVY,TEAL])):ax.bar(x+(j-1)*width,admissions[name]['clusters'],width,label=name,color=color)
 ax.set_xticks(x,[f'C{i}' for i in x]);ax.set_ylabel('Selected images');ax.set_xlabel('Frozen e-means cluster');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
 save(fig,out,'04_admission_environment_coverage','Coverage is not proof of retraining benefit. All arms use the same original replay and fixed optimizer-step endpoint.')
 z=np.load(S/'source_packs_v2.npz');policy=json.loads((S/'protocol/policy_frozen.json').read_text());cluster=np.load(S/policy['cluster']['path']);pool=z['role']=='candidate_pool';labels=cluster['assignment'];distance=cluster['distance'];fig=canvas('Appearance groups: representative original candidate images');gs=fig.add_gridspec(2,4,left=.045,right=.975,top=.855,bottom=.14,hspace=.44,wspace=.10);ids=z['ids'];lookup={r['sample_id']:r for r in rows};chosen=[]
 for k in range(selected['k']):
  indices=np.flatnonzero(pool&(labels==k));i=indices[np.argmin(distance[indices])];r=lookup[str(ids[i])];bgr=cv2.imread(str(Path('/mnt')/r['image']));rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);ax=fig.add_subplot(gs[k//4,k%4]);ax.imshow(rgb);ax.axis('off');ax.set_title(f"C{k} · {len(indices)} candidates\n{WEATHER[int(r['weather'])]} / {int(r['hour']):02d}:00",fontsize=11,color=NAVY);chosen.append(dict(cluster=k,sample_id=r['sample_id'],rgb_sha256=r['expected_sha256']['rgb']))
 save(fig,out,'05_environment_representatives','Closest training-candidate RGB to each final centroid. Weather/time are offline labels shown for interpretation only.')
 (out/'figure_sources.json').write_text(json.dumps(dict(representatives=chosen,roi_example=sample['sample_id'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
 print('SAVED',out,flush=True)
if __name__=='__main__':main()
