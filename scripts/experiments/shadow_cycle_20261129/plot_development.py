"""Static development figures; no independent confirmation claim."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.bridge import sha256_file

def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);d=json.loads(a.summary.read_text());r={x['name']:x for x in d['records']}
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    names=['frozen_RF','frozen_GP','GP_z','RF_rank_plus_z_0.25','RF_rank_plus_z_0.5'];labels=['RF value','GP value','GP contrast z','RF75% + GP z25%','RF50% + GP z50%'];x=np.arange(len(names));w=.36
    fig,ax=plt.subplots(figsize=(11,5.2),layout='constrained');ax.bar(x-w/2,[r[n]['near_error_recall']for n in names],w,label='Near-error capture',color='#426b9c');ax.bar(x+w/2,[r[n]['utility_road_macro']for n in names],w,label='Road-macro utility capture',color='#539a87');ax.set_xticks(x,labels,fontsize=10);ax.set_ylim(0,1);ax.set_ylabel('Recall at fixed10% review budget');ax.set_title('Development comparison: GP contrast helps only as a tested complement');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
    fig.text(.01,.005,'Dev334, parent fit2500, disjoint Meta Cal835. Same budget/audit. Not an independent confirmation or adoption result.',fontsize=9)
    outputs=[]
    for ext in ['png','svg']:
        q=a.output/('01_gp_contrast_development.'+ext);fig.savefig(q,dpi=220,bbox_inches='tight');outputs.append(q)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,kind in zip(axes,['RF','GP']):
        names=[kind+'_meta_'+v for v in ['raw','mean','z']];ax.bar(np.arange(3),[r[n]['near_error_recall']for n in names],color=['#929da9','#6181ad','#5c9c86']);ax.axhline(r['frozen_RF']['near_error_recall'],color='#9b5c49',linestyle='--',label='Frozen RF reference');ax.set_xticks(np.arange(3),['Raw74','+ GP means','+ GP means,z,SD']);ax.set_ylim(0,1);ax.set_title(kind+' meta-input comparison');ax.set_ylabel('Near-error capture at10% budget');ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15)
    fig.suptitle('GP-derived inputs: no arm exceeds the existing frozen RF reference')
    fig.text(.01,.005,'All meta-arms fit the same disjoint835 rows. Parent GP never fits them. Fixed meta-settings; this does not rule out future input recipes.',fontsize=9)
    for ext in ['png','svg']:
        q=a.output/('02_gp_meta_inputs.'+ext);fig.savefig(q,dpi=220,bbox_inches='tight');outputs.append(q)
    plt.close(fig)
    (a.output/'manifest.json').write_text(json.dumps(dict(source_summary_sha256=sha256_file(a.summary),source_sha256=sha256_file(Path(__file__)),artifacts={q.name:sha256_file(q)for q in outputs},scope='Development only, no held-out confirmation adoption claim.'),indent=2)+'\n')
if __name__=='__main__':main()
