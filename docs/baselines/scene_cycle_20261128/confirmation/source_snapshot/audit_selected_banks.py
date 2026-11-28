"""Audit training-bank GT enrichment only after RGB-only admissions are frozen."""
from pathlib import Path
import argparse
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from common.bridge import sha256_file
from quality_estimation.spatial_identity import validate_pack_perception
from quality_estimation.review_targets import support_aware_targets


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pool-packs',type=Path,nargs='+',required=True)
    for n in ['selection','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);selection=json.loads((a.selection/'selection.json').read_text());parts=[]
    for path in a.pool_packs:
        receipt=json.loads(path.with_suffix('.json').read_text());validate_pack_perception(receipt,selection['perception_identity'],'model_state_dict')
        if receipt['sha256']!=sha256_file(path):raise ValueError('selected bank GT pool changed')
        with np.load(path)as z:parts.append({k:z[k]for k in ['ids','counts','weather','hour','site','xy']})
    data={k:np.concatenate([v[k]for v in parts])for k in parts[0]};lookup={s:i for i,s in enumerate(data['ids'])}
    if len(lookup)!=len(data['ids']):raise ValueError('duplicate bank GT sample ID')
    with np.load(a.selection/'feature_only_predictions.npz')as z:
        if z['ids'].tolist()!=data['ids'].tolist():raise ValueError('feature-only prediction / GT pool alignment differs')
        error=z['error_task_scores'];quality=z['estimated_iou']
    target=support_aware_targets(data['counts']);records={}
    def summarize(indices):
        indices=np.asarray(indices,int);q=target['whole_frame_iou'][indices];defined=np.isfinite(q)
        return dict(rows=len(indices),mean_supported_error_utility=float(target['review_utility'][indices].mean()),
            near_error_rows=int(target['near']['error'][indices].sum()),near_fn_rows=int(target['near']['false_negative'][indices].sum()),
            near_fp_rows=int(target['near']['false_positive'][indices].sum()),far_error_rows=int(target['far']['error'][indices].sum()),
            offroad_fp_rows=int(target['offroad']['false_positive'][indices].sum()),near_gt_supported_rows=int((target['near']['gt_pixels'][indices]>=50).sum()),
            mean_defined_lane_iou=float(q[defined].mean())if defined.any()else None,
            spatial_5m_cells=len(np.unique(np.floor(data['xy'][indices]/5).astype(np.int64),axis=0)),
            weather={str(w):int((data['weather'][indices]==w).sum())for w in np.unique(data['weather'])},
            by_weather_utility={str(w):float(target['review_utility'][indices[data['weather'][indices]==w]].mean())for w in np.unique(data['weather'][indices])})
    records['whole_pool']=summarize(np.arange(len(data['ids'])))
    for arm in ['Risk2048','Random2048']:
        admissions=[json.loads(line)for line in (a.selection/(arm+'_rgb_admissions.jsonl')).read_text().splitlines()]
        if any(r['uses_ground_truth']for r in admissions):raise ValueError('bank was not selected GT-free')
        records[arm]=summarize([lookup[r['sample_id']]for r in admissions])
    near=target['near']['error'];auc=float(roc_auc_score(near,error[:,0]))if len(np.unique(near))==2 else None
    q=target['whole_frame_iou'];valid=np.isfinite(q)
    report=dict(records=records,near_event_auc_on_candidate_pool=auc,whole_q_mae_on_candidate_pool=float(np.abs(q[valid]-quality[valid]).mean()),
        risk_minus_matched_random_utility=records['Risk2048']['mean_supported_error_utility']-records['Random2048']['mean_supported_error_utility'],
        perception_identity=selection['perception_identity'],selection_sha256=sha256_file(a.selection/'selection.json'),
        source_pack_sha256={p.name:sha256_file(p)for p in a.pool_packs},source_sha256=sha256_file(Path(__file__)),
        scope='Post-admission GT audit on the opened training pool. Cannot replace independent confirmation or prove perception retraining improvement. Frozen admissions are not retuned by this audit.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');names=['whole_pool','Risk2048','Random2048'];labels=['Pool','Risk','Matched random'];colors=['#9399A3','#347BAB','#B98545']
    axes[0].bar(labels,[records[n]['mean_supported_error_utility']for n in names],color=colors);axes[0].set_ylabel('Mean offline supported-error utility');axes[0].set_title('GT audit after admissions were frozen')
    axes[1].bar(labels,[records[n]['near_error_rows']/records[n]['rows']for n in names],color=colors);axes[1].set_ylabel('Fraction with supported near FN or FP');axes[1].set_title('Candidate GT never used for ranking')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    for ext in ['png','svg']:fig.savefig(a.output/('07_frozen_bank_gt_audit.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig);print('FROZEN BANK GT AUDIT',report['risk_minus_matched_random_utility'],auc,flush=True)


if __name__=='__main__':main()
