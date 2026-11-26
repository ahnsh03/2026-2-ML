"""Illustrate fixed spatial probabilities with hash-bound native image cases.

Choose the median GT-IoU example within each supported spatial class, without
choosing prediction errors. GT is used for offline illustration only.
"""
from pathlib import Path
import argparse,json,hashlib
import cv2,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quality_estimation.drive_calibration import spatial_probabilities


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['extension','data-root','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();e=a.extension;a.output.mkdir(parents=True,exist_ok=False)
    with np.load(e/'evaluation/predictions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    rows={r['sample_id']:r for v in read(e/'registry/index_thinned.json').values() for r in v['rows']}
    replay={r['sample_id']:r for r in map(json.loads,(e/'rgb_replay50_v2/predictions.jsonl').read_text().splitlines())}
    raw_prob=spatial_probabilities(d['raw_mean'],d['raw_covariance'])
    names=['Both retained','Near weak only','Far weak only','Both weak'];records=[]
    for cls in range(4):
        candidates=np.flatnonzero((d['labels']==cls)&(d['near_gt']>=50)&(d['far_gt']>=50))
        if not len(candidates):continue
        candidates=candidates[np.lexsort((d['ids'][candidates],d['truth'][candidates,0]))]
        i=candidates[len(candidates)//2];sid=str(d['ids'][i]);row=rows[sid];r=replay[sid]
        rgbfile=a.data_root/row['image'];gtfile=a.data_root/row['semantic']
        assert sha(rgbfile)==row['expected_sha256']['rgb'] and sha(gtfile)==row['expected_sha256']['semantic']
        rgb=np.array(Image.open(rgbfile).convert('RGB'));gt=np.array(Image.open(gtfile).convert('RGB'))
        heads=e/'rgb_replay50_v2'/r['heads_file'];assert sha(heads)==r['heads_sha256']
        with np.load(heads,allow_pickle=False) as z:mask=cv2.resize(np.isin(z['marking_class'][12:372],[1,2]).astype(np.uint8),(1280,720),interpolation=cv2.INTER_NEAREST)>0
        overlay=rgb.copy();overlay[mask]=(overlay[mask]*.35+np.array([0,255,80])*.65).astype(np.uint8)
        fig=plt.figure(figsize=(16,9));grid=fig.add_gridspec(2,6,height_ratios=[1.2,1],hspace=.35,wspace=.6)
        for j,(im,title) in enumerate(zip([rgb,gt,overlay],['RGB','Own-frame native Semantic GT','Fixed v6 prediction'])):
            ax=fig.add_subplot(grid[0,j*2:j*2+2]);ax.imshow(im);ax.axis('off');ax.set_title(title,fontsize=13)
        ax=fig.add_subplot(grid[1,:3]);x=np.arange(2)
        for offset,key,cov,label,color in [(-.08,'raw_mean','raw_covariance','Raw composed GP','#759ebd'),(.08,'mean','covariance','Fixed v2','#113d67')]:
            mu=d[key][i,1:];sd=np.sqrt(np.diag(d[cov][i])[1:]);ax.errorbar(x+offset,mu,yerr=1.64485362695*sd,fmt='o',capsize=5,label=label,color=color)
        ax.scatter(x,d['truth'][i,1:],marker='x',s=80,c='#ce5a45',label='Offline GT',zorder=10)
        ax.axhline(0,c='#ccc',ls=':');ax.axhline(1,c='#ccc',ls=':');ax.set(xticks=x,xticklabels=['Near-band recall','Far-band recall'],ylabel='Recall; unbounded Gaussian90% intervals');ax.legend(fontsize=10)
        ax=fig.add_subplot(grid[1,3:]);x=np.arange(4)
        ax.bar(x-.18,raw_prob[i],.36,label='Raw',color='#759ebd');ax.bar(x+.18,d['spatial'][i],.36,label='Fixed v2',color='#113d67')
        ax.set(xticks=x,xticklabels=['Both\nretained','Near\nonly weak','Far\nonly weak','Both\nweak'],ylim=(0,1.08),ylabel='Probability');ax.legend(fontsize=10)
        fig.suptitle(f'{names[cls]} | whole IoU GT {d["truth"][i,0]:.3f}, raw GP {np.clip(d["raw_mean"][i,0],0,1):.3f}',fontsize=19)
        fig.text(.035,.03,'Offline GT-selected illustration, not model selection or an operational failure label. Online inputs contain no GT.\nImage bands are not metric depth. Broad intervals express uncertainty; they do not improve point estimates by themselves.',fontsize=11)
        fig.subplots_adjust(top=.88,bottom=.16,left=.06,right=.97)
        fig.savefig(a.output/f'class_{cls}.png',dpi=160);plt.close(fig)
        records.append(dict(sample_id=sid,spatial_class=cls,class_candidates=len(candidates),RGB_sha256=sha(rgbfile),GT_sha256=sha(gtfile),heads_sha256=sha(heads),near_far_GT_pixels=[int(d['near_gt'][i]),int(d['far_gt'][i])]))
    (a.output/'provenance.json').write_text(json.dumps(dict(selection=__doc__,cases=records,missing_classes=[i for i in range(4) if not any(r['spatial_class']==i for r in records)]),indent=2)+'\n')


if __name__=='__main__':main()
