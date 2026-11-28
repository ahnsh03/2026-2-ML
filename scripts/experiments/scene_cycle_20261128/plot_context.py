"""Show direct RGB classification and fixed cluster semantic mappings separately."""
from pathlib import Path
import argparse
import json
import numpy as np
from sklearn.metrics import confusion_matrix,balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.bridge import sha256_file
from quality_estimation.scene_environment import SceneEnvironment,PHOTO_COLUMNS
from quality_estimation.review_targets import support_aware_targets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['environment','pack','histograms','schema','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);env=SceneEnvironment(a.environment);schema=json.loads(a.schema.read_text())
    with np.load(a.pack)as z:data={k:z[k]for k in z.files}
    with np.load(a.histograms)as z:
        if z['ids'].tolist()!=data['ids'].tolist()or z['columns'].tolist()!=env.hist_columns:raise ValueError('context plot RGB alignment differs')
        hist=z['X']
    x=np.c_[hist,data['rgb'][:,[schema['rgb'].index(n)for n in PHOTO_COLUMNS]]];pred=env.predict(x);hour=data['hour']
    targets=dict(weather=data['weather'],lighting=np.where((hour>=8)&(hour<18),0,np.where(((hour>=6)&(hour<8))|((hour>=18)&(hour<20)),1,2)),
                 presence=(support_aware_targets(data['counts'])['scene_gt_class']==2).astype(int))
    labels=dict(weather=['P'+str(i)for i in range(6)],lighting=['day','transition','night'],presence=['near GT absent','near GT present'])
    plt.rcParams.update({'font.family':'DejaVu Sans'});fig,axes=plt.subplots(2,3,figsize=(13,7.7),layout='constrained');records={}
    for column,name in enumerate(['weather','lighting','presence']):
        direct=env.models[name].predict(hist).astype(int);mapped=np.asarray(env.spec['semantic_mapping'][name])[pred['environment_cluster']]
        records[name]={}
        for row,(method,value)in enumerate([('Direct RGB classifier',direct),('Fixed cluster mapping',mapped)]):
            classes=np.arange(len(labels[name]));matrix=confusion_matrix(targets[name],value,labels=classes,normalize='true');score=float(balanced_accuracy_score(targets[name],value));records[name][method]=dict(balanced_accuracy=score,normalized_confusion=matrix.tolist())
            ax=axes[row,column];ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(classes,labels[name],rotation=25,ha='right',fontsize=8);ax.set_yticks(classes,labels[name],fontsize=8)
            ax.set_xlabel('Predicted');ax.set_ylabel('True offline label');ax.set_title(method+' / '+name+'\nBA '+f'{score:.3f}',fontsize=10)
            for i in classes:
                for j in classes:ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white'if matrix[i,j]>.5 else'#243247')
    fig.suptitle('Environment and marking context — all new Dev frames, including low-confidence cases',fontsize=13)
    for ext in ['png','svg']:fig.savefig(a.output/('09_environment_context.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig);report=dict(records=records,rows=len(hist),environment_ood_fraction=float(pred['environment_ood'].mean()),
        environment_manifest_sha256=sha256_file(a.environment/'environment_manifest.json'),source_pack_sha256=sha256_file(a.pack),source_sha256=sha256_file(Path(__file__)),
        scope='Opened Dev illustration. SDK clock categories differ from visible illumination; marking presence is not a normality label. Cluster mappings were fixed on fit labels. No low-confidence frame excluded from the matrix.')
    (a.output/'context_confusion.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
